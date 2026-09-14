"""MGA v0.1: causal multigrid V-cycle transformer, plus full/local baselines.

One V-cycle = bottom-up smoothing + learned pooling (restriction),
full-causal solve at the top level, then top-down cross-attention read
(prolongation) + post-smoothing. The same operator (Block/Pool/Read) is
shared across all levels and all cycles unless `shared=False`.

Causality rule: a summary may be read by token i only if every token it
covers is < i. Tracked explicitly via cover_end per summary (one past the
last covered position, in that level's coordinates), which also makes
shifted partitions and partial blocks sound.

v0.1 additions:
- kq > 1 summaries per block (bandwidth knob; requires b % kq == 0)
- shifted blocking: odd cycles offset the level-0 partition by b//2 so any
  token pair is co-blocked in some cycle (head/mid/tail segmentation keeps
  partial blocks causal-clean)
- posxattn: block-local RoPE on Pool keys; Read q at token index, k at
  cover_end -- the q.k dot product then directly encodes token distance
- fp32read: read softmax computed in fp32 (ignition contrast near the bf16
  noise floor is a suspected cause of the 4096 training failure)
- poolaux: bag-of-words auxiliary loss on level-0 summaries (predict which
  tokens their block contains) -- direct dense gradient to Pool, breaking
  the pool<->read chicken-and-egg that scales with summary count
- full baseline uses SDPA is_causal (flash path on CUDA, no big mask)

CUDA note: folded-block batches can reach bs*nfull = 65536 rows, which
exceeds the CUDA grid y-limit (65535) hit by some SDPA backends. Folded
self-attention is therefore chunked to 8192 rows per call, and CrossAttn
uses manual matmul attention (its matrices are small anyway).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

FOLD_CHUNK = 8192  # max folded-block rows per attention call (grid y-limit)


def causal_mask(t, device):
    return torch.tril(torch.ones(t, t, dtype=torch.bool, device=device))


def run_chunked(fn, x, chunk=FOLD_CHUNK):
    """Apply fn (batch-first) to x in row chunks; keeps CUDA grid dims safe."""
    if x.shape[0] <= chunk:
        return fn(x)
    return torch.cat([fn(xi) for xi in x.split(chunk, dim=0)], dim=0)


class RoPE(nn.Module):
    def __init__(self, hd, max_n):
        super().__init__()
        inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        t = torch.arange(max_n).float()
        f = torch.outer(t, inv)
        self.register_buffer("cos", f.cos(), persistent=False)
        self.register_buffer("sin", f.sin(), persistent=False)

    def rotate(self, x, pos):
        # x: (B, H, T, hd); pos: (T,) long
        cos = self.cos[pos].unsqueeze(0).unsqueeze(0)
        sin = self.sin[pos].unsqueeze(0).unsqueeze(0)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def apply(self, q, k):
        pos = torch.arange(q.shape[-2], device=q.device)
        return self.rotate(q, pos), self.rotate(k, pos)


class Attn(nn.Module):
    def __init__(self, d, h, rope):
        super().__init__()
        self.h = h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.rope = rope

    def forward(self, x, mask):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.h, -1).transpose(1, 2)
        k = k.view(b, t, self.h, -1).transpose(1, 2)
        v = v.view(b, t, self.h, -1).transpose(1, 2)
        q, k = self.rope.apply(q, k)
        if mask is None:  # full causal: flash/mem-efficient path on CUDA
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(o.transpose(1, 2).reshape(b, t, c))


class Block(nn.Module):
    """Pre-norm transformer block: self-attention (masked) + MLP."""

    def __init__(self, d, h, rope, ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = Attn(d, h, rope)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ratio * d), nn.GELU(), nn.Linear(ratio * d, d))

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class CrossAttn(nn.Module):
    """Multi-head cross-attention, manual matmul attention (small matrices;
    avoids SDPA backend quirks and folded-batch grid limits). Keys/values
    carry no positional encoding unless rope + positions are passed.
    fp32_softmax forces the softmax in fp32 under bf16 autocast."""

    def __init__(self, d, h, fp32_softmax=False):
        super().__init__()
        self.h = h
        self.fp32_softmax = fp32_softmax
        self.wq = nn.Linear(d, d, bias=False)
        self.wkv = nn.Linear(d, 2 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, xq, xkv, mask=None, rope=None, q_pos=None, k_pos=None):
        b, tq, c = xq.shape
        tk = xkv.shape[1]
        hd = c // self.h
        q = self.wq(xq).view(b, tq, self.h, hd).transpose(1, 2)
        k, v = self.wkv(xkv).chunk(2, dim=-1)
        k = k.view(b, tk, self.h, hd).transpose(1, 2)
        v = v.view(b, tk, self.h, hd).transpose(1, 2)
        if rope is not None:
            if q_pos is not None:
                q = rope.rotate(q, q_pos)
            if k_pos is not None:
                k = rope.rotate(k, k_pos)
        scores = q @ k.transpose(-2, -1) / math.sqrt(hd)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        if self.fp32_softmax:
            o = torch.softmax(scores.float(), dim=-1).to(v.dtype) @ v
        else:
            o = torch.softmax(scores, dim=-1) @ v
        return self.out(o.transpose(1, 2).reshape(b, tq, c))


class Pool(nn.Module):
    """Restriction: kq learned queries cross-attend to each segment (full
    attention within the segment -- summaries are only read after the segment
    ends, so no leak). Handles partial head/tail segments from shifts and
    non-divisible lengths."""

    def __init__(self, d, h, kq=1, fp32_softmax=False):
        super().__init__()
        self.kq = kq
        self.queries = nn.Parameter(torch.randn(kq, d) * 0.02)
        self.ln = nn.LayerNorm(d)
        self.xattn = CrossAttn(d, h, fp32_softmax)

    def _pool_seg(self, x, rope):
        # x: (B', t, c) -> (B', kq, c)
        q = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        k_pos = torch.arange(x.shape[1], device=x.device) if rope is not None else None
        return run_chunked(lambda xi: self.xattn(
            q[: xi.shape[0]], self.ln(xi), rope=rope, k_pos=k_pos), x)

    def forward(self, x, b, rope=None, shift=0):
        """Returns (summaries (bs, ns, c), cover_end (ns,) long)."""
        bs, t, c = x.shape
        parts, ends = [], []
        if shift > 0:
            parts.append(self._pool_seg(x[:, :shift], rope))
            ends.append(shift)
        nfull = (t - shift) // b
        if nfull:
            mid = x[:, shift:shift + nfull * b].reshape(bs * nfull, b, c)
            o = self._pool_seg(mid, rope)
            parts.append(o.reshape(bs, nfull * self.kq, c))
            ends.extend(shift + (j + 1) * b for j in range(nfull))
        rem = t - shift - nfull * b
        if rem:
            parts.append(self._pool_seg(x[:, t - rem:], rope))
            ends.append(t)
        s = torch.cat(parts, dim=1)
        cover_end = torch.tensor(ends, device=x.device).repeat_interleave(self.kq)
        return s, cover_end


class Read(nn.Module):
    """Prolongation: tokens cross-attend to summaries with cover_end <= their
    position (strictly past coverage), plus an always-visible learned null
    token. Residual, pre-norm. With rope: queries rotate at token index,
    keys at cover_end -- the dot product encodes exact distance."""

    def __init__(self, d, h, fp32_softmax=False):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.xattn = CrossAttn(d, h, fp32_softmax)
        self.null = nn.Parameter(torch.randn(1, 1, d) * 0.02)

    def forward(self, x, summaries, cover_end, rope=None):
        bs, t, c = x.shape
        ns = summaries.shape[1]
        kv = torch.cat([self.null.expand(bs, 1, c), summaries], dim=1)
        i = torch.arange(t, device=x.device).unsqueeze(1)
        allowed = cover_end.unsqueeze(0) <= i  # (t, ns)
        mask = torch.cat(
            [torch.ones(t, 1, dtype=torch.bool, device=x.device), allowed], dim=1
        )
        q_pos = k_pos = None
        if rope is not None:
            q_pos = torch.arange(t, device=x.device)
            k_pos = torch.cat(
                [torch.zeros(1, dtype=torch.long, device=x.device), cover_end]
            )
        return x + self.xattn(self.ln(x), kv, mask, rope=rope, q_pos=q_pos, k_pos=k_pos)


class MGAModel(nn.Module):
    def __init__(self, vocab, n, d=256, h=4, b=16, kq=1, cycles=2, shared=True,
                 posxattn=False, shift=False, fp32read=False, poolaux=False):
        super().__init__()
        assert b % kq == 0, "kq must divide b"
        m, n_levels = n, 0
        while m > b:
            assert m % b == 0, f"sequence length {n} not coarsenable by b={b}"
            m = m * kq // b
            n_levels += 1
        self.b, self.kq, self.cycles, self.shared = b, kq, cycles, shared
        self.posxattn, self.shift = posxattn, (b // 2 if shift else 0)
        self.poolaux = poolaux
        self.vocab = vocab
        self.emb = nn.Embedding(vocab, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight
        self.rope = RoPE(d // h, max(n, 8192))
        if poolaux:
            self.poolaux_head = nn.Linear(d, vocab)
        if shared:
            self.block = Block(d, h, self.rope)
            self.pool = Pool(d, h, kq, fp32read)
            self.read = Read(d, h, fp32read)
        else:
            mods = {}
            for l in range(n_levels):
                mods[f"smooth_{l}"] = Block(d, h, self.rope)
                mods[f"pool_{l}"] = Pool(d, h, kq, fp32read)
                mods[f"read_{l}"] = Read(d, h, fp32read)
                mods[f"post_{l}"] = Block(d, h, self.rope)
            mods["top"] = Block(d, h, self.rope)
            self.mods = nn.ModuleDict(mods)
        self.lnf = nn.LayerNorm(d)

    def _get(self, name):
        if self.shared:
            return {"smooth": self.block, "post": self.block, "top": self.block,
                    "pool": self.pool, "read": self.read}[name.split("_")[0]]
        return self.mods[name]

    def _apply_block(self, blk, s, shift=0):
        """Block-causal smoothing. Full blocks fold into batch (chunked);
        head/tail partial segments processed as their own small blocks."""
        bs, t, c = s.shape
        b = self.b
        if t <= b:
            return blk(s, None)
        parts = []
        if shift > 0:
            parts.append(blk(s[:, :shift], causal_mask(shift, s.device)))
        nfull = (t - shift) // b
        if nfull:
            xb = s[:, shift:shift + nfull * b].reshape(bs * nfull, b, c)
            mask = causal_mask(b, s.device)
            parts.append(run_chunked(lambda xi: blk(xi, mask), xb)
                         .reshape(bs, nfull * b, c))
        rem = t - shift - nfull * b
        if rem:
            parts.append(blk(s[:, t - rem:], causal_mask(rem, s.device)))
        return torch.cat(parts, dim=1)

    def _bow_aux(self, summaries, cover_end, idx):
        """BCE on level-0 summaries predicting the bag-of-words of the block
        they cover. Direct dense gradient to Pool (breaks chicken-and-egg)."""
        bs, t = idx.shape
        ns = cover_end.numel()
        seg_id = torch.bucketize(torch.arange(t, device=idx.device),
                                 cover_end, right=True)
        onehot = F.one_hot(idx, self.vocab).to(summaries.dtype)
        multi = torch.zeros(bs, ns, self.vocab, device=idx.device,
                            dtype=summaries.dtype)
        multi.index_add_(1, seg_id, onehot)
        multi = multi.clamp_(0, 1)
        return F.binary_cross_entropy_with_logits(self.poolaux_head(summaries),
                                                  multi)

    def _vcycle(self, s0, shift=0, idx=None):
        b = self.b
        rope = self.rope if self.posxattn else None
        hier, covers, auxes = [s0], [], []
        s, l = s0, 0
        while s.shape[1] > b:
            sh = shift if l == 0 else 0
            s = self._apply_block(self._get(f"smooth_{l}"), s, sh)
            s, cover_end = self._get(f"pool_{l}")(s, b, rope=rope, shift=sh)
            if l == 0 and idx is not None:
                auxes.append(self._bow_aux(s, cover_end, idx))
            hier.append(s)
            covers.append(cover_end)
            l += 1
        hier[-1] = self._get("top")(hier[-1], None)
        for l in reversed(range(len(hier) - 1)):
            hier[l] = self._get(f"read_{l}")(hier[l], hier[l + 1], covers[l], rope=rope)
            hier[l] = self._apply_block(self._get(f"post_{l}"), hier[l],
                                        shift if l == 0 else 0)
        return hier[0], auxes

    def forward(self, idx, all_cycles=False):
        s = self.emb(idx)
        outs, auxes = [], []
        for t in range(self.cycles):
            s, aux = self._vcycle(s, shift=self.shift if t % 2 == 1 else 0,
                                  idx=idx if self.poolaux else None)
            auxes.extend(aux)
            outs.append(self.head(self.lnf(s)))
        res = outs if all_cycles else outs[-1]
        if self.poolaux:
            return res, (torch.stack(auxes).mean() if auxes else s.sum() * 0.0)
        return res


class BaselineModel(nn.Module):
    """Full-causal transformer (flash path), or block-local stack (receptive
    field = window, invariant to depth) as the local-only baseline."""

    def __init__(self, vocab, n, mode, d=256, h=4, depth=8, window=16):
        super().__init__()
        assert mode in ("full", "local")
        self.mode, self.window = mode, window
        self.emb = nn.Embedding(vocab, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight
        rope = RoPE(d // h, max(n, 8192))
        self.blocks = nn.ModuleList([Block(d, h, rope) for _ in range(depth)])
        self.lnf = nn.LayerNorm(d)

    def forward(self, idx):
        bs, t = idx.shape
        dev = idx.device
        x = self.emb(idx)
        if self.mode == "full":
            for blk in self.blocks:
                x = blk(x, None)  # is_causal inside Attn
        else:
            w = self.window
            xb = x.reshape(bs * (t // w), w, -1)
            mask = causal_mask(w, dev)
            for blk in self.blocks:
                xb = run_chunked(lambda xi: blk(xi, mask), xb)
            x = xb.reshape(bs, t, -1)
        return self.head(self.lnf(x))
