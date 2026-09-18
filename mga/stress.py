"""Stress-test driver: ladder migration with the zero-shot skip rule.

For each rung n in the ladder: evaluate the current checkpoint zero-shot;
if eval_exact >= threshold, skip training and keep the checkpoint; else
train (weights-only resume from the previous rung's checkpoint) and
re-evaluate. Frozen architecture only (--sparse --halo); this script is
pure orchestration.

Usage:
  python -m mga.stress --task mqar --npairs 16
  python -m mga.stress --task passkey
"""
import argparse
import json
import os
import subprocess
import sys


def last_json(stdout):
    lines = [l for l in stdout.splitlines() if l.strip().startswith("{")]
    return json.loads(lines[-1]) if lines else {}


def run(cmd, log_path=None):
    print("[stress]", " ".join(cmd), flush=True)
    if log_path:
        with open(log_path, "w") as lf:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=True)
        return ""
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mqar", choices=["mqar", "passkey"])
    ap.add_argument("--npairs", type=int, default=16)
    ap.add_argument("--nqueries", type=int, default=16)
    ap.add_argument("--ladder", type=int, nargs="+",
                    default=[128, 256, 512, 1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--threshold", type=float, default=0.99,
                    help="zero-shot eval_exact >= threshold -> skip training")
    ap.add_argument("--steps", type=int, default=4000, help="steps per attempt")
    ap.add_argument("--max_steps", type=int, default=12000,
                    help="per-rung cap for top-up training until threshold")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--resume_lr_scale", type=float, default=0.1,
                    help="lr multiplier when resuming a transferred ckpt "
                         "(forgetting guard: L3 lesson — normal lr destroys it)")
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"{args.task}_p{args.npairs}"
    os.makedirs("runs", exist_ok=True)

    def fits(n):
        if args.task != "mqar":
            return True
        return (n + 1 - 3 * args.nqueries) // args.npairs >= 2

    def base_cmd(n):
        cmd = [sys.executable, "-m", "mga.train", "--model", "mga",
               "--task", args.task, "--n", str(n), "--cycles", str(args.cycles),
               "--sparse", "--halo"]
        if args.task == "mqar":
            cmd += ["--nqueries", str(args.nqueries), "--npairs", str(args.npairs)]
        return cmd

    def eval_ckpt(n, ckpt):
        out = run(base_cmd(n) + ["--eval_only", "--resume", ckpt])
        return last_json(out)

    ckpt = None
    rows = []
    for n in args.ladder:
        if not fits(n):
            print(f"[stress] n={n}: pairs don't fit, skipped", flush=True)
            continue
        zs = eval_ckpt(n, ckpt) if ckpt else None
        if zs and zs.get("eval_exact", 0) >= args.threshold:
            print(f"[stress] n={n}: zero-shot {zs['eval_exact']:.3f} "
                  f">= {args.threshold}, skip training", flush=True)
            rows.append((n, zs["eval_exact"], None, "zero-shot"))
            continue
        bs = 64 if n <= 1024 else 32
        lr = args.lr if ckpt is None else args.lr * args.resume_lr_scale
        new_ckpt = f"runs/stress_{tag}_n{n}.pt"
        logf = f"runs/stress_{tag}_n{n}.log"
        total = 0
        while True:
            total += args.steps
            cmd = base_cmd(n) + ["--steps", str(total), "--bs", str(bs),
                                 "--lr", str(lr), "--save", new_ckpt,
                                 "--tag", f"stress_{tag}_n{n}",
                                 "--stop_exact", str(args.threshold)]
            if total > args.steps:
                cmd += ["--resume", new_ckpt]  # top-up: continue own run
            elif ckpt:
                cmd += ["--resume_weights_only", ckpt]  # first attempt: transfer
            run(cmd, logf)
            post = eval_ckpt(n, new_ckpt)
            got = post.get("eval_exact", 0)
            print(f"[stress] n={n}: eval_exact {got:.3f} after {total} steps",
                  flush=True)
            if got >= args.threshold or total >= args.max_steps:
                break
            print(f"[stress] n={n}: below threshold, topping up to "
                  f"{total + args.steps} steps", flush=True)
        ckpt = new_ckpt
        rows.append((n, zs.get("eval_exact") if zs else None, got, "trained"))

    print("\n# summary", flush=True)
    print("| n | zero-shot | trained | action |", flush=True)
    print("|---|---|---|---|", flush=True)
    for n, z, p, act in rows:
        zs_ = f"{z:.3f}" if z is not None else "-"
        ps_ = f"{p:.3f}" if p is not None else "-"
        print(f"| {n} | {zs_} | {ps_} | {act} |", flush=True)


if __name__ == "__main__":
    main()
