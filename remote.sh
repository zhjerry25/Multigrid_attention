#!/usr/bin/env bash
# 远程实验矩阵：n=4096 passkey，v0.1 全配置（exitloss + posxattn + EMA + bf16）
# 用法：bash remote.sh  （建议 tmux/nohup；日志在 runs/*.jsonl）
set -euo pipefail

python -m mga.train --selftest

for T in 1 2 4; do
  python -m mga.train --model mga --n 4096 --cycles $T \
    --steps 8000 --bs 64 --lr 1e-3 --exitloss --posxattn --ema 0.999 --bf16 \
    --save runs/ckpt_4096_T${T}.pt --ckpt_every 1000 \
    --tag remote_4096_T${T} --eval_every 250
done

python -m mga.train --model local --n 4096 --depth 8 \
  --steps 8000 --bs 64 --lr 1e-3 --bf16 \
  --tag remote_4096_local --eval_every 250
