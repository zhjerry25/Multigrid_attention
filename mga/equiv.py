"""Equivalence harness: --dump records reference forward/backward behavior of
MGAModel configs; --check rebuilds from the dumped state_dict and reports
max abs diffs. Run --dump BEFORE any model change, --check after."""
import argparse
import os

import torch
import torch.nn.functional as F

from .model import MGAModel

torch.set_num_threads(1)  # deterministic CPU reductions (bitwise round-trip)

VOCAB, N, D, H, B, KQ = 128, 1024, 64, 2, 16, 1

CONFIGS = {
    "sparse_halo_T2": dict(cycles=2, read_mode="sparse", halo=True),
    "dense_T2": dict(cycles=2, read_mode="dense", halo=False),
    "sparse_halo_T3": dict(cycles=3, read_mode="sparse", halo=True),
}


def build_model(kw, state_dict=None):
    torch.manual_seed(1234)
    model = MGAModel(VOCAB, N, d=D, h=H, b=B, kq=KQ, **kw).eval()
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model


def run(model, idx, tgt):
    out_last = model(idx)
    out_all = model(idx, all_cycles=True)
    loss = F.cross_entropy(out_all[-1].reshape(-1, VOCAB), tgt.reshape(-1))
    model.zero_grad()
    loss.backward()
    grads = torch.cat([p.grad.reshape(-1) for p in model.parameters()
                       if p.grad is not None])
    return out_last, out_all, loss, grads


def dump(path):
    g = torch.Generator().manual_seed(42)
    idx = torch.randint(0, VOCAB, (2, N), generator=g)
    tgt = torch.randint(0, VOCAB, (2, N), generator=g)
    recs = {}
    for name, kw in CONFIGS.items():
        model = build_model(kw)
        out_last, out_all, loss, grads = run(model, idx, tgt)
        recs[name] = dict(state_dict=model.state_dict(), idx=idx, tgt=tgt,
                          out_last=out_last, out_all=out_all, loss=loss,
                          grads=grads)
        print(f"[dump] {name}: loss={loss.item():.6f}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(recs, path)
    print(f"dumped {len(recs)} configs to {path}")


def check(path, tol):
    ref = torch.load(path, map_location="cpu")
    ok_all = True
    for name, kw in CONFIGS.items():
        r = ref[name]
        model = build_model(kw, r["state_dict"])
        out_last, out_all, loss, grads = run(model, r["idx"], r["tgt"])
        assert len(out_all) == len(r["out_all"]), \
            f"{name}: out_all length {len(out_all)} != ref {len(r['out_all'])}"
        diffs = [("out_last", (out_last - r["out_last"]).abs().max().item())]
        diffs += [(f"out_all[{t}]", (a - b).abs().max().item())
                  for t, (a, b) in enumerate(zip(out_all, r["out_all"]))]
        diffs.append(("loss", abs(loss.item() - r["loss"].item())))
        diffs.append(("grads", (grads - r["grads"]).abs().max().item()))
        worst = max(v for _, v in diffs)
        ok = worst <= tol
        ok_all &= ok
        print(f"[{name}] " + "  ".join(f"{k}={v:.3e}" for k, v in diffs))
        print(f"[{name}] worst={worst:.3e} tol={tol:.1e} -> "
              f"{'OK' if ok else 'FAIL'}")
    print(f"overall: {'PASS' if ok_all else 'FAIL'} (tol={tol:.1e})")
    return ok_all


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dump", action="store_true",
                      help="record reference behavior to --path")
    mode.add_argument("--check", action="store_true",
                      help="compare current code against the dumped reference")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--path", default="tmp/equiv_ref.pt")
    args = ap.parse_args()
    if args.dump:
        dump(args.path)
    elif check(args.path, args.tol):
        raise SystemExit(0)
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
