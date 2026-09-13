"""Synthetic H0 tasks: passkey retrieval and copying.

Token layout: 0-9 digits, 10-29 filler, 30=P, 31=Q, 32=SEP. vocab=34.

passkey: filler sequence with a needle [P, d1..d5] at a random position and
[P, Q, d1..d5] at the very end; loss is computed only on the final 5 digits
(the answer). Solving it requires routing one specific distant span to the
end of the sequence -- the make-or-break test for pooling/routing methods.

copying: [pattern c tokens][SEP][filler][SEP][pattern]; loss on last c.
"""
import torch

FILL0, FILL1 = 10, 30
P, Q, SEP = 30, 31, 32
VOCAB = 34
KEY = 5


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
    hi = n + 1 - tail - (KEY + 1)  # needle [P, d1..d5] must not overlap the tail
    pos = torch.randint(0, hi + 1, (bs,), generator=g)
    rows = torch.arange(bs)
    seq[rows, pos] = P
    seq[rows[:, None], pos[:, None] + torch.arange(1, KEY + 1)] = key
    seq[:, -tail] = P
    seq[:, -tail + 1] = Q
    seq[:, -KEY:] = key
    idx, tgt, mask = _targets(seq, KEY)
    return idx.to(device), tgt.to(device), mask.to(device)


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
    return idx.to(device), tgt.to(device), mask.to(device)
