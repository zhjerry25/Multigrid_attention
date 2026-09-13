#!/usr/bin/env bash
# 多卡 sweep launcher：每个配置独占一张卡，卡数不够自动排队。
# 用法:
#   bash sweep.sh            # 自动探测所有 GPU
#   bash sweep.sh 0 1 2 3    # 指定 GPU id
# 日志：runs/sweep_*.jsonl（结构化）+ runs/log_*.txt（stdout）
set -uo pipefail
mkdir -p runs

if [ $# -gt 0 ]; then GPUS=("$@"); else
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || echo 0)
fi
echo "[sweep] GPUs: ${GPUS[*]}"

python -m mga.train --selftest || { echo "[sweep] selftest FAILED, abort"; exit 1; }

MGA="--exitloss --posxattn --ema 0.999 --bf16 --steps 4000 --bs 256 --lr 2e-3 --eval_every 250"
BASE="--bf16 --steps 4000 --bs 256 --lr 1e-3 --eval_every 250"
CURR="--curr 512,1024,2048,4096"

JOBS=()
# H0 主矩阵：passkey × T∈{1,2,4} × kq∈{1,4}
for T in 1 2 4; do for KQ in 1 4; do
  JOBS+=("python -m mga.train --model mga --task passkey --n 4096 --cycles $T --k $KQ --shift $MGA $CURR --save runs/ckpt_T${T}_k${KQ}.pt --ckpt_every 1000 --tag sweep_passkey_T${T}_k${KQ}")
done; done
# 基线（full 走 flash 路径，4096 可跑）
JOBS+=("python -m mga.train --model full --task passkey --n 4096 --depth 8 $BASE --tag sweep_passkey_full")
JOBS+=("python -m mga.train --model local --task passkey --n 4096 --depth 8 $BASE --tag sweep_passkey_local")
# 任务泛化：mqar + copying（T∈{1,2}, kq=1）
for TASK in mqar copying; do for T in 1 2; do
  JOBS+=("python -m mga.train --model mga --task $TASK --n 4096 --cycles $T --k 1 --shift $MGA $CURR --tag sweep_${TASK}_T${T}")
done; done

echo "[sweep] ${#JOBS[@]} jobs on ${#GPUS[@]} GPUs"
i=0
for job in "${JOBS[@]}"; do
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}
  tag=$(echo "$job" | grep -oP '(?<=--tag )\S+')
  # 等该卡空闲（同卡串行，跨卡并行）
  while pgrep -f "CUDA_VISIBLE_DEVICES=${gpu} " >/dev/null 2>&1; do sleep 30; done
  echo "[sweep] GPU $gpu <- $tag"
  CUDA_VISIBLE_DEVICES=$gpu nohup bash -c "$job" > "runs/log_${tag}.txt" 2>&1 &
  i=$((i+1))
  sleep 2
done
wait
echo "[sweep] all done"
