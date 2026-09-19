"""Micro-benchmark: training-step throughput and peak memory for a config.

Used to quantify performance optimizations (e.g. removed redundant
projections). Runs on CPU (sanity) and CUDA (real numbers, bf16 autocast).

Usage:
  python -m mga.bench --model mga --n 4096 --bs 32 --bf16 --sparse --halo --cycles 2 --steps 50
  python -m mga.bench --model full --n 4096 --bs 32 --depth 2 --bf16
  python -m mga.bench --model mga --n 4096 --bs 32 --vocab 50257 --bf16 --sparse --halo
"""
import argparse
import contextlib
import json
import time

import torch
import torch.nn.functional as F

from .model import MGAModel, BaselineModel


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_ctx(args, device):
    if getattr(args, "bf16", False) and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def build(args, device):
    if args.model == "mga":
        m = MGAModel(args.vocab, args.n, d=args.d, h=args.heads, b=args.b,
                     cycles=args.cycles,
                     read_mode="sparse" if args.sparse else "dense",
                     halo=args.halo)
    else:
        m = BaselineModel(args.vocab, args.n, mode=args.model, d=args.d,
                          h=args.heads, depth=args.depth, window=args.b)
    return m.to(device)


def train_step(model, opt, idx, tgt, vocab, ctx):
    with ctx:
        logits = model(idx)
        loss = F.cross_entropy(logits.reshape(-1, vocab), tgt.reshape(-1))
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mga", choices=["mga", "full", "local"])
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--b", type=int, default=16)
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--depth", type=int, default=8, help="baseline depth")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--bf16", action="store_true", help="CUDA bf16 autocast")
    ap.add_argument("--sparse", action="store_true",
                    help="mga: SparseRead (block top-m summaries + fine fanout)")
    ap.add_argument("--halo", action="store_true",
                    help="mga: halo smoothing (prev-block ++ cur-block keys)")
    ap.add_argument("--profile", action="store_true",
                    help="profile 3 extra steps after the timed run")
    args = ap.parse_args()

    device = get_device()
    torch.manual_seed(0)
    model = build(args, device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # fixed synthetic batch, reused every step (throughput measurement only)
    idx = torch.randint(0, args.vocab, (args.bs, args.n), device=device)
    tgt = torch.randint(0, args.vocab, (args.bs, args.n), device=device)
    ctx = amp_ctx(args, device)

    for _ in range(args.warmup):
        train_step(model, opt, idx, tgt, args.vocab, ctx)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        train_step(model, opt, idx, tgt, args.vocab, ctx)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    rec = dict(model=args.model, n=args.n, bs=args.bs, vocab=args.vocab,
               d=args.d)
    if args.model == "mga":
        rec["cycles"] = args.cycles
    rec["params"] = round(n_params / 1e6, 2)
    rec["tok_s"] = int(args.bs * args.n * args.steps / max(elapsed, 1e-9))
    rec["sec"] = round(elapsed, 1)
    if device.type == "cuda":
        rec["peak_mem_mb"] = int(torch.cuda.max_memory_allocated() / 2**20)
    print(json.dumps(rec), flush=True)

    if args.profile:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        sort_key = ("self_cuda_time_total" if device.type == "cuda"
                    else "self_cpu_time_total")
        with torch.profiler.profile(activities=activities) as prof:
            for _ in range(3):
                train_step(model, opt, idx, tgt, args.vocab, ctx)
            if device.type == "cuda":
                torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by=sort_key, row_limit=20))


if __name__ == "__main__":
    main()
