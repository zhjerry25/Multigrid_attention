"""enwik8 byte-level LM data: download, standard split, random-offset batches.

File layout: data/enwik8 (100,000,000 bytes); split convention follows
Transformer-XL et al.: train = first 90M, val = next 5M, test = last 5M.
The whole file is loaded once into RAM (~100MB) and cached per split; no
numpy dependency. lm_batch returns the standard (idx, tgt, mask, pos)
4-tuple with mask covering all positions (dense LM loss).
"""
import os

import torch

URLS = [
    "https://data.deepai.org/enwik8.zip",
    "http://prize.hutter1.net/enwik8.zip",
]
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
FILE = os.path.join(DATA_DIR, "enwik8")
TOTAL = 100_000_000
RANGES = {"train": (0, 90_000_000), "val": (90_000_000, 95_000_000),
          "test": (95_000_000, TOTAL)}

_cache = {}


def _ensure():
    if os.path.exists(FILE):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    import subprocess
    import zipfile
    zpath = FILE + ".zip"
    ok = False
    for url in URLS:
        try:
            subprocess.run(["curl", "-fL", "--retry", "2", "-o", zpath, url],
                           check=True)
            ok = True
            break
        except Exception as e:
            print(f"[lmdata] download failed from {url}: {e}")
    if not ok:
        raise RuntimeError("enwik8 download failed from all mirrors")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(DATA_DIR)
    os.remove(zpath)


def _load(split):
    if split not in _cache:
        _ensure()
        with open(FILE, "rb") as f:
            buf = f.read()
        if buf[:4] == b"PK\x03\x04":  # uploaded the zip instead of the raw file
            import io
            import zipfile
            with zipfile.ZipFile(io.BytesIO(buf)) as z:
                buf = z.read("enwik8")
        if len(buf) != TOTAL:
            raise RuntimeError(
                f"{FILE} has {len(buf):,} bytes, expected {TOTAL:,} -- "
                "truncated upload or wrong file (upload the extracted enwik8)")
        raw = torch.frombuffer(bytearray(buf), dtype=torch.uint8)
        lo, hi = RANGES[split]
        _cache[split] = raw[lo:hi].long()
    return _cache[split]


def lm_batch(bs, n, g, device, split="train"):
    data = _load(split)
    offs = torch.randint(0, len(data) - n - 1, (bs,), generator=g)
    seq = torch.stack([data[o:o + n + 1] for o in offs])
    idx, tgt = seq[:, :-1], seq[:, 1:]
    mask = torch.ones(bs, n, dtype=torch.bool)
    return idx.to(device), tgt.to(device), mask.to(device), None
