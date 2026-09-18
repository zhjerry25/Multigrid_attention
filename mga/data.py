"""Synthetic H0 tasks: passkey, copying, MQAR (multi-query associative recall).

Token layout: 0-9 digits, 10-29 filler, 30=P, 31=Q, 32=SEP. vocab=34.
All batchers return (idx, tgt, mask, pos): idx/tgt are the LM-shifted pair
(model input length n), mask marks loss positions in tgt, pos is the needle
position per sample (None for non-passkey tasks) for depth-bucketed eval.

passkey: filler with a needle [P, d1..d5] at a random position and
[P, Q, d1..d5] at the end; loss on the final 5 digits.

copying: [pattern c tokens][SEP][filler][SEP][pattern]; loss on last c.

mqar: n_pairs key->value pairs [(k_i, v_i)] spaced through filler, then
[Q, k, v] x n_queries at the end; loss on the query values. Keys are 64
dedicated tokens (34-97) sampled WITHOUT replacement (a permutation per
sequence) -- an earlier version drew pairs with replacement from too few
keys, making targets contradictory (same key, different values) and
freezing the loss at ln(10). Fillers are 26-29 for this task only
(passkey/copying keep 18-29); values are digits.
"""
import torch

FILL0, FILL1 = 18, 30  # filler tokens 18..29 (passkey/copying)
P, Q, SEP = 30, 31, 32
VOCAB = 128  # 0-9 digits, 18-29 filler, 30-32 P/Q/SEP, 34-97 MQAR keys
KEY = 5
MQAR_KEYS = list(range(34, 98))  # 64 distinct keys (npairs <= 64)
MQAR_FILL0, MQAR_FILL1 = 26, 30  # mqar-only fillers (26..29)


def _targets(seq, loss_len):
    idx, tgt = seq[:, :-1], seq[:, 1:]
    mask = torch.zeros(seq.shape[0], seq.shape[1] - 1, dtype=torch.bool)
    mask[:, -loss_len:] = True
    return idx, tgt, mask


def passkey_batch(bs, n, g, device):
    # seq has n+1 tokens so that idx = seq[:-1] keeps the model input at n.
    seq = torch.randint(FILL0, FILL1, (bs, n + 1), generator=g)
    key = torch.randint(0, 10, (bs, KEY), generator=g)
    tail = KEY + 2  # [P, Q, d1..d5] at the end
    hi = n + 1 - tail - (KEY + 1)  # needle must not overlap the tail
    pos = torch.randint(0, hi + 1, (bs,), generator=g)
    rows = torch.arange(bs)
    seq[rows, pos] = P
    seq[rows[:, None], pos[:, None] + torch.arange(1, KEY + 1)] = key
    seq[:, -tail] = P
    seq[:, -tail + 1] = Q
    seq[:, -KEY:] = key
    idx, tgt, mask = _targets(seq, KEY)
    return idx.to(device), tgt.to(device), mask.to(device), pos.to(device)


def copying_batch(bs, n, g, device, c=None):
    c = c or n // 8
    assert 2 * c + 2 <= n + 1
    seq = torch.randint(FILL0, FILL1, (bs, n + 1), generator=g)
    pat = torch.randint(FILL0, FILL1, (bs, c), generator=g)
    seq[:, :c] = pat
    seq[:, c] = SEP
    seq[:, -c - 1] = SEP
    seq[:, -c:] = pat
    idx, tgt, mask = _targets(seq, c)
    return idx.to(device), tgt.to(device), mask.to(device), None


def mqar_batch(bs, n, g, device, n_pairs=16, n_queries=4):
    assert n_pairs <= len(MQAR_KEYS), \
        f"n_pairs={n_pairs} > {len(MQAR_KEYS)} distinct keys"
    tail = 3 * n_queries
    seq = torch.randint(MQAR_FILL0, MQAR_FILL1, (bs, n + 1), generator=g)
    seg = (n + 1 - tail) // n_pairs
    assert seg >= 2, "sequence too short for n_pairs"
    ki = torch.argsort(torch.rand(bs, len(MQAR_KEYS), generator=g), dim=1)[:, :n_pairs]
    keys = MQAR_KEYS[0] + ki  # unique keys per sequence (no-replacement)
    vals = torch.randint(0, 10, (bs, n_pairs), generator=g)
    off = torch.randint(0, seg - 1, (bs, n_pairs), generator=g)
    p = torch.arange(n_pairs).unsqueeze(0) * seg + off  # (bs, n_pairs)
    rows = torch.arange(bs).unsqueeze(1)
    seq[rows, p] = keys
    seq[rows, p + 1] = vals
    qi = torch.randint(0, n_pairs, (bs, n_queries), generator=g)
    t = n + 1 - tail
    seq[:, t::3] = Q
    seq[:, t + 1::3] = keys.gather(1, qi)
    seq[:, t + 2::3] = vals.gather(1, qi)
    idx, tgt = seq[:, :-1], seq[:, 1:]
    mask = torch.zeros(bs, n, dtype=torch.bool)
    mask[:, t + 1::3] = True  # tgt positions predicting each value (after [Q,k])
    return idx.to(device), tgt.to(device), mask.to(device), None
