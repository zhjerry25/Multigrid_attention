"""MGA v0: causal multigrid V-cycle transformer, plus full/local baselines.

One V-cycle = bottom-up smoothing + learned pooling (restriction),
full-causal solve at the top level, then top-down cross-attention read
(prolongation) + post-smoothing. The same operator (Block/Pool/Read) is
shared across all levels and all cycles unless `shared=False`.

Causality rule: the summary of block j covers tokens [j*b, (j+1)*b) and may
only be read by tokens in blocks with index > j (strictly past). A learned
null key/value is always readable so early blocks have a well-defined read.

posxattn=True adds explicit position to the cross-attention paths (v0.1):
block-local RoPE on Pool keys and block-index RoPE on Read queries/keys, so
order and distance are directly representable instead of being smuggled
inside token content.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(t, device):
    return torch.tril(torch.ones(t, t, dtype=torch.bool, device=device))


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
    """Multi-head cross-attention. By default keys/values carry no positional
    encoding; pass rope + positions to make order/distance explicit."""

    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.wq = nn.Linear(d, d, bias=False)
        self.wkv = nn.Linear(d, 2 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, xq, xkv, mask=None, rope=None, q_pos=None, k_pos=None):
        b, tq, c = xq.shape
        tk = xkv.shape[1]
        q = self.wq(xq).view(b, tq, self.h, -1).transpose(1, 2)
        k, v = self.wkv(xkv).chunk(2, dim=-1)
        k = k.view(b, tk, self.h, -1).transpose(1, 2)
        v = v.view(b, tk, self.h, -1).transpose(1, 2)
        if rope is not None:
            if q_pos is not None:
                q = rope.rotate(q, q_pos)
            if k_pos is not None:
                k = rope.rotate(k, k_pos)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(o.transpose(1, 2).reshape(b, tq, c))


class Pool(nn.Module):
    """Restriction: kq learned queries cross-attend to each block (full within
    block -- the summary is only ever read after the block ends, so no leak).
    With rope, keys get block-local positions so order is addressable."""

    def __init__(self, d, h, kq=1):
        super().__init__()
        self.kq = kq
        self.queries = nn.Parameter(torch.randn(kq, d) * 0.02)
        self.ln = nn.LayerNorm(d)
        self.xattn = CrossAttn(d, h)

    def forward(self, x, b, rope=None):
        bs, t, c = x.shape
        nb = t // b
        xb = self.ln(x).reshape(bs * nb, b, c)
        q = self.queries.unsqueeze(0).expand(bs * nb, -1, -1)
        k_pos = torch.arange(b, device=x.device) if rope is not None else None
        o = self.xattn(q, xb, rope=rope, k_pos=k_pos)
        return o.reshape(bs, nb * self.kq, c)


class Read(nn.Module):
    """Prolongation: tokens cross-attend to summaries of strictly-past blocks
    (plus an always-visible learned null token). Residual, pre-norm. With
    rope, queries rotate by own block index and keys by summary block index,
    so relative block distance is explicit."""

    def __init__(self, d, h):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.xattn = CrossAttn(d, h)
        self.null = nn.Parameter(torch.randn(1, 1, d) * 0.02)

    def forward(self, x, summaries, b, rope=None):
        bs, t, c = x.shape
        ns = summaries.shape[1]
        kv = torch.cat([self.null.expand(bs, 1, c), summaries], dim=1)
        i = torch.arange(t, device=x.device).unsqueeze(1)
        js = torch.arange(ns, device=x.device).unsqueeze(0)
        mask = torch.cat(
            [torch.ones(t, 1, dtype=torch.bool, device=x.device), js < (i // b)],
            dim=1,
        )
        q_pos = k_pos = None
        if rope is not None:
            q_pos = torch.arange(t, device=x.device) // b
            k_pos = torch.cat(
                [torch.zeros(1, dtype=torch.long, device=x.device),
                 torch.arange(ns, device=x.device)]
            )
        return x + self.xattn(self.ln(x), kv, mask, rope=rope, q_pos=q_pos, k_pos=k_pos)


class MGAModel(nn.Module):
    def __init__(self, vocab, n, d=256, h=4, b=16, kq=1, cycles=2, shared=True,
                 posxattn=False):
        super().__init__()
        assert kq == 1, "v0 supports kq=1 only (bandwidth knob is v0.1)"
        m, n_levels = n, 0
        while m > b:
            assert m % b == 0, f"sequence length {n} not coarsenable by b={b}"
            m //= b
            n_levels += 1
        self.b, self.cycles, self.shared = b, cycles, shared
        self.posxattn = posxattn
        self.emb = nn.Embedding(vocab, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight
        self.rope = RoPE(d // h, max(n, 8192))
        if shared:
            self.block = Block(d, h, self.rope)
            self.pool = Pool(d, h, kq)
            self.read = Read(d, h)
        else:
            mods = {}
            for l in range(n_levels):
                mods[f"smooth_{l}"] = Block(d, h, self.rope)
                mods[f"pool_{l}"] = Pool(d, h, kq)
                mods[f"read_{l}"] = Read(d, h)
                mods[f"post_{l}"] = Block(d, h, self.rope)
            mods["top"] = Block(d, h, self.rope)
            self.mods = nn.ModuleDict(mods)
        self.lnf = nn.LayerNorm(d)

    def _get(self, name):
        if self.shared:
            return {"smooth": self.block, "post": self.block, "top": self.block,
                    "pool": self.pool, "read": self.read}[name.split("_")[0]]
        return self.mods[name]

    def _apply_block(self, blk, s):
        """Block-causal smoothing, computed by folding blocks into the batch
        dimension (avoids materializing a huge masked attention matrix)."""
        bs, t, c = s.shape
        b = self.b
        if t <= b:
            return blk(s, causal_mask(t, s.device))
        xb = s.reshape(bs * (t // b), b, c)
        return blk(xb, causal_mask(b, s.device)).reshape(bs, t, c)

    def _vcycle(self, s0):
        b = self.b
        rope = self.rope if self.posxattn else None
        hier = [s0]
        s, l = s0, 0
        while s.shape[1] > b:
            s = self._apply_block(self._get(f"smooth_{l}"), s)
            s = self._get(f"pool_{l}")(s, b, rope=rope)
            hier.append(s)
            l += 1
        hier[-1] = self._apply_block(self._get("top"), hier[-1])
        for l in reversed(range(len(hier) - 1)):
            hier[l] = self._get(f"read_{l}")(hier[l], hier[l + 1], b, rope=rope)
            hier[l] = self._apply_block(self._get(f"post_{l}"), hier[l])
        return hier[0]

    def forward(self, idx, all_cycles=False):
        s = self.emb(idx)
        if all_cycles:
            outs = []
            for _ in range(self.cycles):
                s = self._vcycle(s)
                outs.append(self.head(self.lnf(s)))
            return outs
        for _ in range(self.cycles):
            s = self._vcycle(s)
        return self.head(self.lnf(s))


class BaselineModel(nn.Module):
    """Full-causal transformer, or block-local stack (receptive field = window,
    invariant to depth) as the local-only baseline. Same Block throughout."""

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
            mask = causal_mask(t, dev)
            for blk in self.blocks:
                x = blk(x, mask)
        else:
            w = self.window
            xb = x.reshape(bs * (t // w), w, -1)
            mask = causal_mask(w, dev)
            for blk in self.blocks:
                xb = blk(xb, mask)
            x = xb.reshape(bs, t, -1)
        return self.head(self.lnf(x))
