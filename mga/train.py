"""Train/eval driver for MGA and baselines.

Usage:
  python -m mga.train --selftest
  python -m mga.train --model mga --n 4096 --cycles 2 --k 4 --shift \
      --steps 4000 --bs 256 --lr 1e-3 --exitloss --posxattn --ema 0.999 \
      --bf16 --fp32read --poolaux --curr 512,1024,2048,4096 \
      --save runs/ck.pt --ckpt_every 1000
  python -m mga.train --model local --n 4096
  python -m mga.train --model full --n 4096 --bf16   # flash path on CUDA
"""
import argparse
import contextlib
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from . import data
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


def build(args, n=None, d=None, device="cpu"):
    n = n or args.n
    d = d or args.d
    if args.model == "mga":
        m = MGAModel(data.VOCAB, n, d=d, h=args.heads, b=args.b, kq=args.k,
                     cycles=args.cycles, shared=not args.unshared,
                     posxattn=getattr(args, "posxattn", False),
                     shift=getattr(args, "shift", False),
                     fp32read=getattr(args, "fp32read", False),
                     poolaux=getattr(args, "poolaux", False))
    else:
        m = BaselineModel(data.VOCAB, n, mode=args.model, d=d, h=args.heads,
                          depth=args.depth, window=args.b)
    return m.to(device)


def coarsenable(n, b, kq):
    m = n
    while m > b:
        if m % b != 0:
            return False
        m = m * kq // b
    return True


BATCHERS = {"passkey": data.passkey_batch, "copying": data.copying_batch,
            "mqar": data.mqar_batch}


def make_batch(args, g, device, n=None):
    return BATCHERS[args.task](args.bs, n or args.n, g, device)


def loss_and_acc(model, idx, tgt, mask, exit_w=None, aux_w=0.0):
    """CE on masked positions; with exit_w (mga only), weighted sum of
    per-cycle exit losses; with poolaux, models return (logits, aux) and the
    BOW auxiliary loss is added with weight aux_w. Accuracy is always from
    the last cycle. Returns (loss, digit_acc, exact, hit_vec)."""
    ret = model(idx, all_cycles=exit_w is not None)
    aux = ret[1] if isinstance(ret, tuple) else None
    logits = ret[0] if isinstance(ret, tuple) else ret
    if exit_w is not None:
        loss = sum(w * F.cross_entropy(lg[mask], tgt[mask])
                   for w, lg in zip(exit_w, logits)) / sum(exit_w)
        logits = logits[-1]
    else:
        loss = F.cross_entropy(logits[mask], tgt[mask])
    if aux is not None and aux_w > 0:
        loss = loss + aux_w * aux
    lp, tp = logits[mask], tgt[mask]
    hit = (lp.argmax(-1) == tp).view(idx.shape[0], -1)
    hit_vec = hit.all(1)
    return loss, hit.float().mean().item(), hit_vec.float().mean().item(), hit_vec


def save_ckpt(path, model, opt, ema, args, step):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "ema": ema, "args": vars(args), "step": step}, path)


def leak_test():
    """Shuffle the last 10% of input tokens; logits over the first 90% must
    not change. Run on CPU for determinism. Covers shift/kq>1/posxattn too --
    the single most important test in this codebase."""
    n, cutoff = 1024, 1024 - 102
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, data.VOCAB, (2, n), generator=g)
    idx2 = idx.clone()
    idx2[:, cutoff:] = torch.randint(0, data.VOCAB, (2, n - cutoff), generator=g)
    variants = [
        ("mga", dict(model="mga", k=1, shift=False, posxattn=False)),
        ("mga_shift_k4_posx", dict(model="mga", k=4, shift=True, posxattn=True)),
        ("full", dict(model="full", k=1, shift=False, posxattn=False)),
        ("local", dict(model="local", k=1, shift=False, posxattn=False)),
    ]
    ok_all = True
    for name, v in variants:
        args = argparse.Namespace(n=n, d=64, heads=2, b=16, cycles=2,
                                  unshared=False, depth=4, **v)
        model = build(args, device="cpu").eval()
        with torch.no_grad():
            l1, l2 = model(idx), model(idx2)
        err = (l1[:, :cutoff] - l2[:, :cutoff]).abs().max().item()
        ok = err < 1e-4
        ok_all &= ok
        print(f"[leak] {name:16s} max-logit-diff on past positions: {err:.2e} "
              f"-> {'OK' if ok else 'CAUSAL LEAK!'}")
    assert ok_all, "causal leak detected; do not train until fixed"


def overfit_test(device):
    """A single fixed batch (n=512) must be memorizable: T=2 MGA should reach
    ~100% exact match within a few hundred steps."""
    args = argparse.Namespace(model="mga", n=512, d=128, heads=4, b=16, k=1,
                              cycles=2, unshared=False, depth=4, task="passkey",
                              bs=32, shift=False, posxattn=False)
    torch.manual_seed(0)
    model = build(args, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(1)
    idx, tgt, mask, _ = make_batch(args, g, device)
    model.train()
    for step in range(400):
        loss, pd, em, _ = loss_and_acc(model, idx, tgt, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == 399:
            print(f"[overfit] step {step:4d} loss {loss.item():.4f} "
                  f"digit-acc {pd:.3f} exact {em:.3f}")
    print("[overfit] done (expect exact ~1.0)")


def lr_at(step, total, base):
    warm = max(1, total // 20)
    if step < warm:
        return base * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def train(args, device):
    torch.manual_seed(args.seed)
    curr = None
    if args.curr:
        curr = [int(x) for x in args.curr.split(",")]
        for n_c in curr:
            assert coarsenable(n_c, args.b, args.k), f"curr length {n_c} not coarsenable"
        print(f"length curriculum {curr} for first half of training", flush=True)
    model = build(args, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={args.model} n={args.n} params={n_params/1e6:.2f}M "
          f"cycles={getattr(args, 'cycles', '-')} device={device}", flush=True)
    exit_w = None
    if args.exitloss and args.model == "mga":
        t = args.cycles
        exit_w = [1.0] if t == 1 else [0.3 + 0.7 * i / (t - 1) for i in range(t)]
        print(f"exit loss weights: {[round(w, 2) for w in exit_w]}", flush=True)
    aux_w = 0.1 if (args.poolaux and args.model == "mga") else 0.0
    if aux_w:
        print(f"pool BOW aux loss weight {aux_w}", flush=True)
    ema = None
    if args.ema > 0:
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if ema is not None and ck.get("ema") is not None:
            ema = ck["ema"]
        start_step = ck["step"] + 1
        print(f"resumed from {args.resume} at step {start_step}", flush=True)
    if args.eval_only:
        assert args.resume, "--eval_only requires --resume"
        if ck.get("ema") is not None:
            model.load_state_dict(ck["ema"])
        model.eval()
        g_eval = torch.Generator().manual_seed(args.seed + 200)
        ems, pds, hits, poss = [], [], [], []
        with torch.no_grad(), amp_ctx(args, device):
            for _ in range(8):
                idx, tgt, mask, pos = make_batch(args, g_eval, device)
                _, pd, em, hv = loss_and_acc(model, idx, tgt, mask)
                ems.append(em)
                pds.append(pd)
                if pos is not None:
                    hits.append(hv.cpu())
                    poss.append(pos.cpu())
        rec = dict(n=args.n, cycles=getattr(args, "cycles", None),
                   eval_exact=round(sum(ems) / len(ems), 4),
                   eval_digit=round(sum(pds) / len(pds), 4))
        if poss:
            h, p = torch.cat(hits).float(), torch.cat(poss)
            rec["depth_exact"] = [
                round(h[(p >= qi * args.n // 4) & (p < (qi + 1) * args.n // 4)]
                      .mean().item(), 3)
                if ((p >= qi * args.n // 4) & (p < (qi + 1) * args.n // 4)).any()
                else -1
                for qi in range(4)
            ]
        if args.model == "mga":
            idx, tgt, mask, _ = make_batch(args, g_eval, device)
            with torch.no_grad(), amp_ctx(args, device):
                lgs = model(idx, all_cycles=True)
            lgs = lgs[0] if isinstance(lgs, tuple) else lgs
            rec["cycles_exact"] = [
                round((lg[mask].argmax(-1) == tgt[mask]).view(idx.shape[0], -1)
                      .all(1).float().mean().item(), 3)
                for lg in lgs
            ]
        print(json.dumps(rec), flush=True)
        return
    g_train = torch.Generator().manual_seed(args.seed + 100)
    g_eval = torch.Generator().manual_seed(args.seed + 200)
    os.makedirs("runs", exist_ok=True)
    log = open(f"runs/{args.tag}.jsonl", "a")
    t0, last_t = time.time(), time.time()
    for step in range(start_step, args.steps):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step, args.steps, args.lr)
        n_b = args.n
        if curr is not None and step < args.steps // 2:
            n_b = curr[torch.randint(len(curr), (1,), generator=g_train).item()]
        idx, tgt, mask, pos = make_batch(args, g_train, device, n=n_b)
        model.train()
        with amp_ctx(args, device):
            loss, pd, em, _ = loss_and_acc(model, idx, tgt, mask, exit_w, aux_w)
        opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ema is not None:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    ema[k].mul_(args.ema).add_(v.detach(), alpha=1 - args.ema)
        if step % 100 == 0:
            now = time.time()
            tok_s = int(args.bs * n_b * 100 / max(now - last_t, 1e-9)) if step > 0 else 0
            last_t = now
            rec = dict(step=step, loss=round(loss.item(), 4), digit_acc=round(pd, 4),
                       exact=round(em, 4), tok_s=tok_s, sec=round(now - t0, 1),
                       gnorm=round(float(gnorm), 3),
                       emb_n=round(model.emb.weight.norm().item(), 3))
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if args.save and args.ckpt_every and (step + 1) % args.ckpt_every == 0:
            save_ckpt(args.save, model, opt, ema, args, step)
        if step % args.eval_every == 0 or step == args.steps - 1:
            backup = None
            if ema is not None:
                backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema)
            model.eval()
            ems, pds, hits, poss = [], [], [], []
            with torch.no_grad(), amp_ctx(args, device):
                for _ in range(4):
                    idx, tgt, mask, pos = make_batch(args, g_eval, device)
                    _, pd, em, hv = loss_and_acc(model, idx, tgt, mask)
                    ems.append(em)
                    pds.append(pd)
                    if pos is not None:
                        hits.append(hv.cpu())
                        poss.append(pos.cpu())
                rec = dict(step=step, eval_exact=round(sum(ems) / len(ems), 4),
                           eval_digit=round(sum(pds) / len(pds), 4))
                if poss:
                    h, p = torch.cat(hits).float(), torch.cat(poss)
                    depth = []
                    for qi in range(4):
                        m2 = (p >= qi * args.n // 4) & (p < (qi + 1) * args.n // 4)
                        depth.append(round(h[m2].mean().item(), 3) if m2.any() else -1)
                    rec["depth_exact"] = depth
                if args.model == "mga":
                    idx, tgt, mask, _ = make_batch(args, g_eval, device)
                    lgs = model(idx, all_cycles=True)
                    lgs = lgs[0] if isinstance(lgs, tuple) else lgs
                    rec["cycles_exact"] = [
                        round((lg[mask].argmax(-1) == tgt[mask]).view(idx.shape[0], -1)
                              .all(1).float().mean().item(), 3)
                        for lg in lgs
                    ]
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if backup is not None:
                model.load_state_dict(backup)
    if args.save:
        save_ckpt(args.save, model, opt, ema, args, args.steps - 1)
        print(f"saved {args.save}", flush=True)
    log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mga", choices=["mga", "full", "local"])
    ap.add_argument("--task", default="passkey", choices=["passkey", "copying", "mqar"])
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--b", type=int, default=16)
    ap.add_argument("--k", type=int, default=1, help="summaries per block (bandwidth)")
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--depth", type=int, default=8, help="baseline depth")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--ema", type=float, default=0.0, help="EMA eval decay, 0=off")
    ap.add_argument("--bf16", action="store_true", help="CUDA bf16 autocast")
    ap.add_argument("--fp32read", action="store_true",
                    help="v0.1: force read-path softmax in fp32 under bf16")
    ap.add_argument("--poolaux", action="store_true",
                    help="v0.1: BOW auxiliary loss on level-0 summaries")
    ap.add_argument("--curr", default="", help="length curriculum, e.g. 512,1024,4096")
    ap.add_argument("--save", default="", help="checkpoint path (.pt)")
    ap.add_argument("--ckpt_every", type=int, default=0, help="periodic save interval")
    ap.add_argument("--resume", default="", help="resume from checkpoint path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unshared", action="store_true")
    ap.add_argument("--shift", action="store_true",
                    help="v0.1: shifted blocking on odd cycles (b//2 offset)")
    ap.add_argument("--posxattn", action="store_true",
                    help="v0.1: positional encoding on Pool/Read cross-attention")
    ap.add_argument("--exitloss", action="store_true",
                    help="v0.1: per-cycle exit loss with increasing weights (mga only)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--eval_only", action="store_true",
                    help="load --resume checkpoint, eval once at args.n/--cycles, exit")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.tag is None:
        args.tag = f"{args.model}_{args.task}_n{args.n}_T{args.cycles}_s{args.seed}"
    device = get_device()
    if args.selftest:
        leak_test()
        overfit_test(device)
        return
    train(args, device)


if __name__ == "__main__":
    main()
