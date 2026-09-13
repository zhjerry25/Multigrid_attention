"""Synthetic H0 tasks: passkey, copying, MQAR (multi-query associative recall).

Token layout: 0-9 digits, 10-29 filler, 30=P, 31=Q, 32=SEP. vocab=34.
All batchers return (idx, tgt, mask, pos): idx/tgt are the LM-shifted pair
(model input length n), mask marks loss positions in tgt, pos is the needle
position per sample (None for non-passkey tasks) for depth-bucketed eval.

passkey: filler with a needle [P, d1..d5] at a random position and
[P, Q, d1..d5] at the end; loss on the final 5 digits.

copying: [pattern c tokens][SEP][filler][SEP][pattern]; loss on last c.

mqar: n_pairs key->value pairs [(k_i, v_i)] spaced through filler, query
[Q, k_q, v_q] at the end; loss on the final value. Keys are 8 dedicated
tokens (22-29 shared with filler range is avoided: keys 10-17), values are
digits -- recall requires content-addressed lookup among many distractors.
"""
import torch

FILL0, FILL1 = 18, 30  # filler tokens 18..29 (10-17 reserved as MQAR keys)
P, Q, SEP = 30, 31, 32
VOCAB = 34
KEY = 5
MQAR_KEYS = list(range(10, 18))


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


def mqar_batch(bs, n, g, device, n_pairs=16):
    seq = torch.randint(FILL0, FILL1, (bs, n + 1), generator=g)
    seg = (n + 1 - 3) // n_pairs
    assert seg >= 2, "sequence too short for n_pairs"
    ki = torch.randint(0, len(MQAR_KEYS), (bs, n_pairs), generator=g)
    keys = torch.tensor(MQAR_KEYS)[ki]  # (bs, n_pairs)
    vals = torch.randint(0, 10, (bs, n_pairs), generator=g)
    off = torch.randint(0, seg - 1, (bs, n_pairs), generator=g)
    p = torch.arange(n_pairs).unsqueeze(0) * seg + off  # (bs, n_pairs)
    rows = torch.arange(bs).unsqueeze(1)
    seq[rows, p] = keys
    seq[rows, p + 1] = vals
    qi = torch.randint(0, n_pairs, (bs,), generator=g)
    r = torch.arange(bs)
    seq[:, -3] = Q
    seq[r, -2] = keys[r, qi]
    seq[r, -1] = vals[r, qi]
    idx, tgt, mask = _targets(seq, 1)
    return idx.to(device), tgt.to(device), mask.to(device), None
