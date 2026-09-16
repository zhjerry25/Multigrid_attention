"""MGA: causal multigrid V-cycle transformer.

One V-cycle = halo block-causal smoothing (keys = prev ++ current block)
+ learned pooling (restriction) up a regular summary tree + full-causal
solve at the top + top-down read (prolongation) + post-smoothing, repeated
for T cycles. A single operator (Block/Pool/Read) is shared across all
levels and all cycles unless `shared=False`; parameters do not depend on
sequence length, which is what enables zero-shot length transfer.

Causality: a summary may be read by token i only if every token it covers
is < i, tracked via cover_end per summary (one past the last covered
position, in that level's coordinates). Any change to an attention path
must keep the leak test at 0 diff.

Level-0 read modes (`read_mode`): "dense" (all past summaries),
"sparse" (production: block-shared global top-m selection + top-mf
raw-block fanout, O(n*m) read), "amr" (rejected tree-descent variant,
kept for audit). Other rejected audit flags (default off, verdicts in
topic.md): kq>1, shift, posxattn, fp32read, poolaux, exitloss.

CUDA note: folded-block batches are chunked to FOLD_CHUNK rows per
attention call (some SDPA backends exceed the CUDA grid y-limit at
65536 rows), and CrossAttn uses manual matmul attention.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

FOLD_CHUNK = 8192  # max folded-block rows per attention call (grid y-limit)


def causal_mask(t, device):
    return torch.tril(torch.ones(t, t, dtype=torch.bool, device=device))


def halo_mask(b, device):
    """Halo window (2b): in-block query q attends keys k <= b + q (its own
    position in the window). Previous block is fully visible."""
    i = torch.arange(b, device=device).unsqueeze(1)
    j = torch.arange(2 * b, device=device).unsqueeze(0)
    return j <= (b + i)


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

    def forward_halo(self, q_x, kv_x, mask):
        """Attention with queries from q_x and keys/values from a longer
        window kv_x (prev block ++ current block). RoPE: keys at window
        positions 0..tk-1, queries at tk-tq..tk-1 (relative offsets exact)."""
        b, tq, c = q_x.shape
        tk = kv_x.shape[1]
        q = self.qkv(q_x).chunk(3, dim=-1)[0]
        k, v = self.qkv(kv_x).chunk(3, dim=-1)[1:]
        q = q.view(b, tq, self.h, -1).transpose(1, 2)
        k = k.view(b, tk, self.h, -1).transpose(1, 2)
        v = v.view(b, tk, self.h, -1).transpose(1, 2)
        dev = q_x.device
        q = self.rope.rotate(q, torch.arange(tk - tq, tk, device=dev))
        k = self.rope.rotate(k, torch.arange(tk, device=dev))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(o.transpose(1, 2).reshape(b, tq, c))


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

    def forward_halo(self, q_x, kv_x, mask):
        """Same block, but attention queries come from q_x and keys/values
        from the wider window kv_x (shared projections, shared LN)."""
        x = q_x + self.attn.forward_halo(self.ln1(q_x), self.ln1(kv_x), mask)
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
    """Prolongation (dense): tokens cross-attend to summaries with
    cover_end <= their position, plus an always-visible learned null token.
    Residual, pre-norm."""

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


class AMRRead(nn.Module):
    """REJECTED (audit only): tree-descent read. Greedy descent let the
    scorer's learning be held hostage by its own selections (beam myopia);
    SparseRead's global selection won. Kept behind read_mode="amr".

    Batched top-down tree descent + top-m fine fanout.

    Per token: b*log_b(n) beam keys + m*b fine keys -> O(n*b*log n) total.
    Requires kq=1, no shift, >=2 summary levels (regular b-ary tree).
    Selection indices are detached; gradients flow through softmax weights
    and the fine read. Selected gathers get a straight-through correction
    (value unchanged, gradient routes the fine path's success back to the
    scorer's selected arm). Training-time Gumbel noise (tau annealed to 0
    by the trainer) gives cold-start exploration; eval is always hard."""

    def __init__(self, d, h, m=4):
        super().__init__()
        self.h, self.m = h, m
        self.tau = 0.0  # Gumbel noise scale, annealed externally
        self.ln = nn.LayerNorm(d)
        self.scorer = CrossAttn(d, h)  # beam scoring, shared across levels
        self.fine = CrossAttn(d, h)    # fine read of selected raw tokens
        self.null = nn.Parameter(torch.randn(1, 1, d) * 0.02)

    def _beam_attn(self, q, cand, mask):
        """q: (bs,h,n,hd); cand: (bs,n,c,d) per-token candidates; mask:
        (bs,n,c) bool or None. Returns (ctx (bs,n,d), head-mean probs (bs,n,c))."""
        bs, n, c, d = cand.shape
        hd = q.shape[-1]
        k, v = self.scorer.wkv(cand).chunk(2, dim=-1)
        k = k.view(bs, n, c, self.h, hd)
        v = v.view(bs, n, c, self.h, hd)
        scores = torch.einsum("bhnd,bnchd->bhnc", q, k) / math.sqrt(hd)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        w = torch.softmax(scores, dim=-1)
        ctx = torch.einsum("bhnc,bnchd->bnhd", w, v).reshape(bs, n, d)
        return ctx, w.mean(dim=1)

    def _select(self, probs):
        """argmax with optional Gumbel exploration noise (train only)."""
        if self.training and self.tau > 0:
            u = torch.rand_like(probs).clamp_min(1e-9)
            probs = probs + (-torch.log(-u.log())) * self.tau
        return probs.argmax(dim=-1)

    @staticmethod
    def _st(gathered, probs, sel):
        """Straight-through correction: forward value unchanged, backward
        sends gathered-content gradient to the selected arm's probability."""
        if sel.dim() == probs.dim() - 1:
            p = probs.gather(-1, sel.unsqueeze(-1).clamp_min(0))
        else:
            p = probs.gather(-1, sel.clamp_min(0))
        while p.dim() < gathered.dim():
            p = p.unsqueeze(-1)
        return gathered * (1.0 + (p - p.detach()))

    def forward(self, x, up_levels, covers):
        """x: (bs,n,d) level-0 states; up_levels: [hier[1]..hier[L]] (>=2);
        covers kept for interface parity (cover ranges follow the tree)."""
        bs, n, d = x.shape
        b = n // up_levels[0].shape[1]  # block factor of the regular tree
        L = len(up_levels)
        top = up_levels[-1]
        ntop = top.shape[1]
        q = self.scorer.wq(self.ln(x)).view(bs, n, self.h, -1).transpose(1, 2)

        # initial candidates: null + top summaries, masked by cover_end <= i
        cand = torch.cat([self.null.expand(bs, 1, d), top], dim=1)
        cand = cand.unsqueeze(1).expand(bs, n, 1 + ntop, d)
        cover_top = (torch.arange(ntop, device=x.device) + 1) * (n // ntop)
        i = torch.arange(n, device=x.device).unsqueeze(1)
        mask = torch.cat(
            [torch.ones(n, 1, dtype=torch.bool, device=x.device), cover_top <= i],
            dim=1,
        ).unsqueeze(0).expand(bs, n, 1 + ntop)

        ctx, parent = 0, None
        topm, topm_probs = None, None
        for lvl in range(L - 1, -1, -1):
            step_ctx, probs = self._beam_attn(q, cand, mask)
            ctx = ctx + step_ctx
            sel = self._select(probs)  # (bs, n), detached (noisy) argmax
            if lvl == 0:
                topm = probs.topk(min(self.m, probs.shape[-1]), dim=-1).indices
                topm_probs = probs
            else:
                # gather children of the selected summary as next candidates
                child_src = up_levels[lvl - 1]
                groups = child_src.view(bs, -1, b, d)
                null_group = self.null.expand(bs, b, d).unsqueeze(1)
                groups = torch.cat([null_group, groups], dim=1)
                cand = groups[torch.arange(bs).unsqueeze(1), sel]
                cand = self._st(cand, probs, sel)
                mask = None  # children tile parent's past range: always safe
                parent = sel

        # fine fanout: expand the top-m level-1 summaries to level-0 blocks.
        # block_map index: 0 = null block; 1+g = level-0 block g.
        ns1 = up_levels[0].shape[1]
        null_block = self.null.expand(bs, b, d).unsqueeze(1)
        block_map = torch.cat([null_block, x.view(bs, ns1, b, d)], dim=1)
        base = torch.where(parent == 0,
                           torch.zeros_like(parent),
                           (parent - 1) * b + 1)
        fine_idx = torch.where((parent == 0).unsqueeze(-1),
                               torch.zeros_like(topm),
                               base.unsqueeze(-1) + topm)  # (bs, n, m)
        fine_keys = block_map[torch.arange(bs).view(bs, 1, 1), fine_idx]
        fine_keys = self._st(fine_keys, topm_probs, topm)
        fine_keys = fine_keys.reshape(bs, n, -1, d)

        qf = self.fine.wq(self.ln(x)).view(bs, n, self.h, -1).transpose(1, 2)
        fk, fv = self.fine.wkv(fine_keys).chunk(2, dim=-1)
        hd = qf.shape[-1]
        fk = fk.view(bs, n, fine_keys.shape[2], self.h, hd)
        fv = fv.view(bs, n, fine_keys.shape[2], self.h, hd)
        scores = torch.einsum("bhnd,bnchd->bhnc", qf, fk) / math.sqrt(hd)
        w = torch.softmax(scores, dim=-1)
        fctx = torch.einsum("bhnc,bnchd->bnhd", w, fv).reshape(bs, n, d)
        return x + self.fine.out(ctx + fctx)


class SparseRead(nn.Module):
    """Prolongation (sparse): block-shared global top-m summary selection +
    top-mf fine fanout. Selection is global and recomputed every step (free
    exploration, MoBA-style); only the read is sparse -> O(n*m + n*mf*b).
    Falls back to the dense Read as m -> n/b. Causality: a block's candidate
    summaries must satisfy cover_end <= block start (same rule as dense)."""

    def __init__(self, d, h, m=64, mf=4):
        super().__init__()
        self.h, self.m, self.mf = h, m, mf
        self.ln = nn.LayerNorm(d)
        self.scorer = CrossAttn(d, h)
        self.fine = CrossAttn(d, h)
        self.null = nn.Parameter(torch.randn(1, 1, d) * 0.02)

    def forward(self, x, summaries, cover_end):
        bs, n, d = x.shape
        ns = summaries.shape[1]
        b = n // ns  # block factor (level-0 block size)
        nb = ns
        # keys/values: null + summaries, block-representative scores
        kv = torch.cat([self.null.expand(bs, 1, d), summaries], dim=1)
        K, V = self.scorer.wkv(kv).chunk(2, dim=-1)  # (bs, ns+1, d)
        q = self.scorer.wq(self.ln(x))  # (bs, n, d)
        qblk = q.view(bs, nb, b, d).mean(dim=2)  # (bs, nb, d) block rep
        # causal selection: block j's selection is computed from block j-1's
        # mean query (all <= j*b <= i). A within-block mean would leak future
        # tokens' queries into earlier tokens' key selection.
        qrep = torch.cat([qblk[:, :1], qblk[:, :-1]], dim=1)
        scores = qrep @ K.transpose(-2, -1) / math.sqrt(d)  # (bs, nb, ns+1)
        j = torch.arange(nb, device=x.device).unsqueeze(1)
        allowed = torch.cat(
            [torch.ones(nb, 1, dtype=torch.bool, device=x.device),
             cover_end.unsqueeze(0) <= (j * b)], dim=1)  # (nb, ns+1)
        scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
        m = min(self.m, ns + 1)
        idx = scores.topk(m, dim=-1).indices  # (bs, nb, m), sorted desc
        valid = allowed.unsqueeze(0).expand(bs, -1, -1).gather(-1, idx)  # (bs,nb,m)
        # gather selected keys/values, block-wise attention (no per-token
        # materialization)
        gK = K[torch.arange(bs).view(bs, 1, 1), idx]  # (bs, nb, m, d)
        gV = V[torch.arange(bs).view(bs, 1, 1), idx]
        hd = d // self.h
        qt = q.view(bs, nb, b, self.h, hd)
        gKh = gK.view(bs, nb, m, self.h, hd)
        gVh = gV.view(bs, nb, m, self.h, hd)
        att = torch.einsum("bqzhd,bqmhd->bqhzm", qt, gKh) / math.sqrt(hd)
        att = att.masked_fill(~valid.unsqueeze(2).unsqueeze(2), float("-inf"))
        w = torch.softmax(att, dim=-1)
        ctx = torch.einsum("bqhzm,bqmhd->bqzhd", w, gVh).reshape(bs, n, d)
        # fine fanout: top-mf of the selection expand to their raw blocks.
        # block_map: 0 = null block, 1+g = raw block g (matches idx: 0=null,
        # s>=1 -> block s-1)
        mf = min(self.mf, m)
        fidx = idx[:, :, :mf]  # (bs, nb, mf)
        fvalid = valid[:, :, :mf]  # blocks with <mf valid past summaries
        # may have selected masked (invalid) entries whose raw blocks could be
        # future blocks -- mask them out of the fine attention
        fmask = fvalid.unsqueeze(-1).expand(-1, -1, -1, b).reshape(bs, nb, mf * b)
        block_map = torch.cat([self.null.expand(bs, b, d).unsqueeze(1),
                               x.view(bs, nb, b, d)], dim=1)
        fkeys = block_map[torch.arange(bs).view(bs, 1, 1), fidx]  # (bs,nb,mf,b,d)
        fkeys = fkeys.reshape(bs, nb, mf * b, d)
        fK, fV = self.fine.wkv(fkeys).chunk(2, dim=-1)
        fKh = fK.view(bs, nb, mf * b, self.h, hd)
        fVh = fV.view(bs, nb, mf * b, self.h, hd)
        qf = self.fine.wq(self.ln(x)).view(bs, nb, b, self.h, hd)
        fatt = torch.einsum("bqzhd,bqmhd->bqhzm", qf, fKh) / math.sqrt(hd)
        fatt = fatt.masked_fill(~fmask.unsqueeze(2).unsqueeze(2), float("-inf"))
        fw = torch.softmax(fatt, dim=-1)
        fctx = torch.einsum("bqhzm,bqmhd->bqzhd", fw, fVh).reshape(bs, n, d)
        return x + self.fine.out(ctx + fctx)


class MGAModel(nn.Module):
    def __init__(self, vocab, n, d=256, h=4, b=16, kq=1, cycles=2, shared=True,
                 posxattn=False, shift=False, fp32read=False, poolaux=False,
                 read_mode="dense", amr_m=4, read_m=64, read_mf=4, halo=False):
        super().__init__()
        assert b % kq == 0, "kq must divide b"
        if read_mode == "amr":
            assert kq == 1 and not shift, "AMR read requires kq=1 and no shift"
        m, n_levels = n, 0
        while m > b:
            assert m % b == 0, f"sequence length {n} not coarsenable by b={b}"
            m = m * kq // b
            n_levels += 1
        if read_mode == "amr":
            assert n_levels >= 2, f"AMR read needs >=2 summary levels (n={n} too small)"
        assert not (halo and shift), "halo + shift unsupported"
        self.b, self.kq, self.cycles, self.shared = b, kq, cycles, shared
        self.posxattn, self.shift = posxattn, (b // 2 if shift else 0)
        self.poolaux = poolaux
        self.read_mode = read_mode
        self.halo = halo
        self.vocab = vocab
        self.emb = nn.Embedding(vocab, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight
        self.rope = RoPE(d // h, max(n, 8192))
        if poolaux:
            self.poolaux_head = nn.Linear(d, vocab)
        if read_mode == "amr":
            self.amrread = AMRRead(d, h, m=amr_m)
        elif read_mode == "sparse":
            self.sparseread = SparseRead(d, h, m=read_m, mf=read_mf)
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

    def _apply_block_halo(self, blk, s):
        """Halo smoothing: keys = previous block ++ current block (2b window).
        Block 0 runs plain block-causal (no past exists)."""
        bs, t, c = s.shape
        b = self.b
        if t <= b:
            return blk(s, None)
        xb = s.view(bs, -1, b, c)
        first = blk(xb[:, 0], causal_mask(b, s.device)).unsqueeze(1)
        if xb.shape[1] == 1:
            return first.view(bs, t, c)
        rest, prev = xb[:, 1:], xb[:, :-1]
        kv = torch.cat([prev, rest], dim=2).reshape(bs * rest.shape[1], 2 * b, c)
        q = rest.reshape(bs * rest.shape[1], b, c)
        mask = halo_mask(b, s.device)
        outs = [blk.forward_halo(qi, kvi, mask)
                for qi, kvi in zip(q.split(FOLD_CHUNK), kv.split(FOLD_CHUNK))]
        out = torch.cat(outs, dim=0)
        return torch.cat([first, out.view(bs, -1, b, c)], dim=1).view(bs, t, c)

    def _smooth(self, blk, s, shift=0):
        if self.halo:
            return self._apply_block_halo(blk, s)
        return self._apply_block(blk, s, shift)

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
            s = self._smooth(self._get(f"smooth_{l}"), s, sh)
            s, cover_end = self._get(f"pool_{l}")(s, b, rope=rope, shift=sh)
            if l == 0 and idx is not None:
                auxes.append(self._bow_aux(s, cover_end, idx))
            hier.append(s)
            covers.append(cover_end)
            l += 1
        hier[-1] = self._get("top")(hier[-1], None)
        for l in reversed(range(len(hier) - 1)):
            if l == 0 and self.read_mode == "amr":
                hier[0] = self.amrread(hier[0], hier[1:], covers)
            elif l == 0 and self.read_mode == "sparse":
                hier[0] = self.sparseread(hier[0], hier[1], covers[0])
            else:
                hier[l] = self._get(f"read_{l}")(hier[l], hier[l + 1],
                                                covers[l], rope=rope)
            hier[l] = self._smooth(self._get(f"post_{l}"), hier[l],
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
