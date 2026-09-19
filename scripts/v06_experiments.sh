#!/bin/bash
# v0.6 experiment batch: sanity + Phase B (B1 qdelta / B2 split_roles / B3 cycle_unshared)
# Runs entirely on the v0.6 branch tip — no checkouts needed.
# Logs: training -> runs/<tag>.jsonl; gate/diag evals -> runs/gate_*.json, runs/diag_*.json
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

echo "=== [0/4] sanity: frozen arch on v0.6 tip (expect 512 ignite ~0.95, 4096 zero-shot ~0.96) ==="
$MGA --task passkey --n 512  --cycles 2 --sparse --halo --steps 1500 --bs 64 --lr 1e-3 --save runs/pk_v06_512.pt  --tag pk_v06_512
$MGA --task passkey --n 4096 --cycles 2 --sparse --halo --eval_only --resume runs/pk_v06_512.pt | tee runs/gate_v06.json

echo "=== [1/4] B1: --qdelta (coarse-context rewrite of fine query) ==="
$MGA --task passkey --n 512  --cycles 2 --sparse --halo --qdelta --steps 3000 --bs 64 --lr 1e-3 --save runs/pk_qdelta512.pt --tag pk_qdelta512
$MGA --task passkey --n 4096 --cycles 2 --sparse --halo --qdelta --eval_only --resume runs/pk_qdelta512.pt --diag | tee runs/gate_qdelta.json
if gate runs/gate_qdelta.json 0.5 B1; then
  $MGA --task lm --n 4096 --cycles 2 --sparse --halo --qdelta --steps 15000 --bs 32 --lr 1e-3 --save runs/lm_qdelta.pt --tag lm_qdelta
  $MGA --task lm --n 4096 --cycles 2 --sparse --halo --qdelta --eval_only --resume runs/lm_qdelta.pt --diag | tee runs/diag_qdelta.json
fi

echo "=== [2/4] B2: --split_roles (pre/post/top split) + d=352 param-fair control ==="
$MGA --task passkey --n 512  --cycles 2 --sparse --halo --split_roles --steps 3000 --bs 64 --lr 1e-3 --save runs/pk_roles512.pt --tag pk_roles512
$MGA --task passkey --n 4096 --cycles 2 --sparse --halo --split_roles --eval_only --resume runs/pk_roles512.pt --diag | tee runs/gate_roles.json
if gate runs/gate_roles.json 0.5 B2; then
  $MGA --task lm --n 4096 --cycles 2 --sparse --halo --split_roles --steps 15000 --bs 32 --lr 1e-3 --save runs/lm_roles.pt --tag lm_roles
  $MGA --task lm --n 4096 --cycles 2 --sparse --halo --d 352 --steps 15000 --bs 32 --lr 1e-3 --save runs/lm_d352.pt --tag lm_d352
  $MGA --task lm --n 4096 --cycles 2 --sparse --halo --split_roles --eval_only --resume runs/lm_roles.pt --diag | tee runs/diag_roles.json
fi

echo "=== [3/4] B3: --cycle_unshared (per-cycle operator sets) ==="
$MGA --task passkey --n 512  --cycles 2 --sparse --halo --cycle_unshared --steps 3000 --bs 64 --lr 1e-3 --save runs/pk_cycu512.pt --tag pk_cycu512
$MGA --task passkey --n 4096 --cycles 2 --sparse --halo --cycle_unshared --eval_only --resume runs/pk_cycu512.pt --diag | tee runs/gate_cycu.json
if gate runs/gate_cycu.json 0.5 B3; then
  $MGA --task lm --n 4096 --cycles 2 --sparse --halo --cycle_unshared --steps 15000 --bs 32 --lr 1e-3 --save runs/lm_cycu.pt --tag lm_cycu
  $MGA --task lm --n 4096 --cycles 2 --sparse --halo --cycle_unshared --eval_only --resume runs/lm_cycu.pt --diag | tee runs/diag_cycu.json
fi

echo "=== v0.6 batch done ==="
