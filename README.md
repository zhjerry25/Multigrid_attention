# Multigrid Attention（MGA）

层级 × 循环的线性注意力：把全注意力的完全图换成直径 2·log_b(n) 的树状层级图，top-m 稀疏读取，所有层级与循环共享同一算子。**当前状态：v0.4——线性化不掉点成立，真实语言上反超参数匹配基线。**

## 快速开始

```bash
python3 -m venv .venv && .venv/bin/pip install torch
.venv/bin/python -m mga.train --selftest   # 泄漏测试（必须全 0 diff）+ 过拟合测试
```

enwik8 LM 需要 `data/enwik8`（100,000,000 字节，自行放入；支持直接放 zip）。

## 仓库地图

| 路径 | 内容 |
|---|---|
| `mga/model.py` | MGAModel（V-cycle：平滑→池化→递归→延拓→后平滑，T 循环全共享权重）、SparseRead（top-m 摘要 + fine fanout）、halo 平滑、full/local 基线、泄漏审计 flag（AMR/shift/exitloss/posxattn/poolaux，均已否决，留档） |
| `mga/data.py` | 合成任务：passkey / copying / MQAR（keys 无放回） |
| `mga/lmdata.py` | enwik8 下载/切分/批次（90M/5M/5M） |
| `mga/train.py` | 训练/评估驱动：`--sparse --halo --k --cycles --read_m --read_mf --resume --resume_weights_only --eval_only --curr --bf16 --ema --save` |
| `sweep.sh` | 多卡 launcher（per-GPU worker） |
| `topic.md` | **项目大脑**：方法定义、全部结果、根因理论、否决清单、文献定位、标度/KV cache 分析 |
| `paper/outline.md` | 论文骨架（故事弧 + 图表清单 + 审稿人预答） |
| `runs/*.jsonl` | 结构化实验日志（gnorm/emb_n/depth_exact/cycles_exact/posloss_mod_b） |

## 核心结果速览（详见 topic.md）

- passkey：512 点火 → 4096 不掉点（0.9609）→ 16k 反超 dense（0.8984）→ **65k 零样本 0.9375**
- MQAR：128 阶梯点火 → 512 迁移 1.0 → **4096 零样本 1.0（step 0）**
- **enwik8 char-LM @4096：bpc 1.5557，反超参数匹配 full-d2（1.664）**，posloss 平坦
- 标度：read/显存严格线性，选择 O(n²/b²)（交叉点 ~4M）；KV cache 同阶 T 层 transformer；生成 O(1)/token

## 生产配方

点火（任何任务）→ 迁移（任何尺度）：

```bash
# 1. 小尺度点火（SNR 越阈即可，如 512，MQAR 用 128）
python -m mga.train --model mga --task passkey --n 512 --cycles 2 --sparse \
  --steps 3000 --bs 64 --lr 1e-3 --save runs/stage512.pt
# 2. 目标尺度上岗（零样本或续训）
python -m mga.train --model mga --task passkey --n 4096 --cycles 2 --sparse \
  --resume_weights_only runs/stage512.pt --eval_only
# LM 直接冷启动即可（稠密监督天然点火），记得 --halo
python -m mga.train --model mga --task lm --n 4096 --cycles 2 --sparse --halo \
  --steps 15000 --bs 32 --lr 1e-3
```
