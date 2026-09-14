#!/usr/bin/env bash
# 多卡 sweep：任务 round-robin 分给各卡，每卡一个串行 worker——同一卡上
# 任意时刻只有一个训练进程，结构上杜绝抢显存（不用 pgrep 轮询）。
# 用法:
#   bash sweep.sh            # 自动探测所有 GPU
#   bash sweep.sh 0 2 3      # 指定 GPU id
#   PYTHON=.venv/bin/python bash sweep.sh   # 指定解释器（默认 python）
# 日志：runs/sweep_*.jsonl（结构化）+ runs/log_<tag>.txt（每任务 stdout）
set -uo pipefail
mkdir -p runs

PY=${PYTHON:-python}
if [ $# -gt 0 ]; then GPUS=("$@"); else
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || echo 0)
fi
NG=${#GPUS[@]}
echo "[sweep] GPUs: ${GPUS[*]} (python: $PY)"

$PY -m mga.train --selftest || { echo "[sweep] selftest FAILED, abort"; exit 1; }

MGA="--ema 0.999 --bf16 --steps 4000 --bs 256 --lr 1e-3 --eval_every 250"
BASE="--bf16 --steps 4000 --bs 256 --lr 1e-3 --eval_every 250"
CURR="--curr 512,1024,2048,4096"

TAGS=(); JOBS=()
add() { TAGS+=("$1"); JOBS+=("$2"); }

# H0 主矩阵：passkey × T∈{1,2,4} × kq∈{1,4}
for T in 1 2 4; do for KQ in 1 4; do
  tag="sweep_passkey_T${T}_k${KQ}"
  add "$tag" "$PY -m mga.train --model mga --task passkey --n 4096 --cycles $T --k $KQ --shift $MGA $CURR --save runs/ckpt_T${T}_k${KQ}.pt --ckpt_every 1000 --tag $tag"
done; done
# 基线（full 走 flash 路径，4096 可跑）
add "sweep_passkey_full"  "$PY -m mga.train --model full  --task passkey --n 4096 --depth 8 $BASE --tag sweep_passkey_full"
add "sweep_passkey_local" "$PY -m mga.train --model local --task passkey --n 4096 --depth 8 $BASE --tag sweep_passkey_local"
# 任务泛化：mqar + copying（T∈{1,2}, kq=1）
for TASK in mqar copying; do for T in 1 2; do
  tag="sweep_${TASK}_T${T}"
  add "$tag" "$PY -m mga.train --model mga --task $TASK --n 4096 --cycles $T --k 1 --shift $MGA $CURR --tag $tag"
done; done

echo "[sweep] ${#JOBS[@]} jobs on $NG GPU(s)"
for gi in "${!GPUS[@]}"; do
  gpu=${GPUS[$gi]}
  script="runs/worker_gpu${gpu}.sh"
  {
    echo "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    for ((j=gi; j<${#JOBS[@]}; j+=NG)); do
      echo "echo '[worker $gpu] start ${TAGS[$j]}'"
      echo "CUDA_VISIBLE_DEVICES=$gpu ${JOBS[$j]} > runs/log_${TAGS[$j]}.txt 2>&1"
      echo "echo '[worker $gpu] done  ${TAGS[$j]}'"
    done
  } > "$script"
  nohup bash "$script" > "runs/log_gpu${gpu}.txt" 2>&1 &
  echo "[sweep] GPU $gpu worker pid $! ($((${#JOBS[@]} / NG + (${#JOBS[@]} % NG > gi))) jobs)"
done
wait
echo "[sweep] all done"
