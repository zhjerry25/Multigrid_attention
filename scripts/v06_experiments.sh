#!/bin/bash
# v0.6 experiment batch: sanity + Phase B (B1 qdelta / B2 split_roles / B3 cycle_unshared)
# Runs entirely on the v0.6 branch tip — no checkouts needed.
#
# RECIPE NOTE (verified 2026-09-19): sparse passkey 512->4096 ZERO-SHOT transfer
# fails (~0.1) on ANY code version, old or new — at n=512 the candidate count
# (ns+1 = 33) <= m=64, so top-m selection is a no-op during ignition and the
# selector's ranking is never trained. The historical recipe ALWAYS continued
# training at 4096 with a fresh cosine (--resume_weights_only; S2' hit 0.96).
# Gates below therefore use: ignite@512 -> continue@4096 (fresh cosine) -> eval.
#
# Logs: training -> runs/<tag>.jsonl; gate/diag evals -> runs/gate_*.json,
# runs/diag_*.json
set -u
cd "$(dirname "$0")/.."

# enwik8 symlink for worktree setups (main repo keeps the real data/)
if [ ! -e data/enwik8 ] && [ -e ~/autodl-tmp/Multigrid_attention/data/enwik8 ]; then
  ln -sfn ~/autodl-tmp/Multigrid_attention/data data
  echo "[setup] symlinked data/ from main repo"
fi

MGA="python -m mga.train --model mga"

gate() {  # $1=eval jsonl, $2=threshold, $3=label -> nonzero exit if below threshold
  python - "$1" "$2" "$3" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith("{")]
ex = [r["eval_exact"] for r in rows if "eval_exact" in r]
ok = bool(ex) and ex[-1] >= float(sys.argv[2])
print(f"[gate] {sys.argv[3]}: eval_exact={ex[-1] if ex else 'N/A'} "
      f"threshold={sys.argv[2]} -> {'PASS' if ok else 'SKIP LM RUN'}", flush=True)
sys.exit(0 if ok else 1)
EOF
}

# per-variant passkey pipeline: ignite@512 -> continue@4096 (fresh cosine) -> gate+diag
# usage: pk_gate <label> [extra model flags...]
pk_gate() {
  local V=$1; shift
  $MGA --task passkey --n 512  --cycles 2 --sparse --halo "$@" --steps 3000 --bs 64 --lr 1e-3 --seed 0 \
       --save runs/pk_${V}_512.pt --tag pk_${V}_512
  $MGA --task passkey --n 4096 --cycles 2 --sparse --halo "$@" --steps 3000 --bs 32 --lr 1e-3 \
       --resume_weights_only runs/pk_${V}_512.pt --save runs/pk_${V}_4096.pt --tag pk_${V}_4096
  $MGA --task passkey --n 4096 --cycles 2 --sparse --halo "$@" --eval_only --resume runs/pk_${V}_4096.pt --diag \
       | tee runs/gate_${V}.json
  gate runs/gate_${V}.json 0.5 $V
}

echo "=== [0/4] sanity: frozen arch, historical S-recipe (expect ~0.9 after 4096 continuation) ==="
pk_gate v06

echo "=== [1/4] B1: --qdelta (coarse-context rewrite of fine query) ==="
if pk_gate qdelta --qdelta; then
  $MGA --task lm --n 4096 --cycles 2 --sparse --halo --qdelta --steps 15000 --bs 32 --lr 1e-3 --save runs/lm_qdelta.pt --tag lm_qdelta
  $MGA --task lm --n 4096 --cycles 2 --sparse --halo --qdelta --eval_only --resume runs/lm_qdelta.pt --diag | tee runs/diag_qdelta.json
fi

echo "=== [2/4] B2: --split_roles (pre/post/top split) + d=352 param-fair control ==="
if pk_gate roles --split_roles; then
  $MGA --task lm --n 4096 --cycles 2 --sparse --halo --split_roles --steps 15000 --bs 32 --lr 1e-3 --save runs/lm_roles.pt --tag lm_roles
  $MGA --task lm --n 4096 --cycles 2 --sparse --halo --d 352 --steps 15000 --bs 32 --lr 1e-3 --save runs/lm_d352.pt --tag lm_d352
  $MGA --task lm --n 4096 --cycles 2 --sparse --halo --split_roles --eval_only --resume runs/lm_roles.pt --diag | tee runs/diag_roles.json
fi

echo "=== [3/4] B3: --cycle_unshared (per-cycle operator sets) ==="
if pk_gate cycu --cycle_unshared; then
  $MGA --task lm --n 4096 --cycles 2 --sparse --halo --cycle_unshared --steps 15000 --bs 32 --lr 1e-3 --save runs/lm_cycu.pt --tag lm_cycu
  $MGA --task lm --n 4096 --cycles 2 --sparse --halo --cycle_unshared --eval_only --resume runs/lm_cycu.pt --diag | tee runs/diag_cycu.json
fi

echo "=== v0.6 batch done ==="
