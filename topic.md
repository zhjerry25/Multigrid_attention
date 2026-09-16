# Multigrid Attention（MGA）：层级 × 循环的线性注意力

**状态：v0.4 收官（2026-09-15）。一句话：把全注意力的完全图换成直径 2·log_b(n) 的树状层级图，top-m 稀疏读取，全部层级与循环共享同一算子——用循环深度和层级路由换注意力范围。**

## 方法（生产形态）

输入 x ∈ R^{n×d}，块 b=16。共享算子：Θ（pre-norm transformer block）、Pool（learned cross-attn 池化）、Read。一个 **V-cycle**：

1. **平滑**：halo 块因果注意力（keys = 前一块 ++ 本块，2b 窗口；块折进 batch 维计算）
2. **限制**：每块池化出 1 个摘要（learned query cross-attn）
3. **递归**：同一 Θ 重复至顶层（≤ b 个），顶层全因果注意力
4. **延拓**：level-0 用 **SparseRead**；高层（≥1）用稠密 read
5. **后平滑**；共 T=2 个 cycle，权重全共享

**SparseRead**（核心部件）：每个 query 块（用前一块均值 query，保因果）对全部摘要打分选 top-m=64（(n/b)² 选择成本，每步全局重打分 = 免费探索）；块内 token 读取这些摘要（O(n·m)）；m 中 top-mf=4 展开为其覆盖的原始 token 做 token 级直连（无效选择掩码）。m→n/b 退化为稠密 read。

**因果规则**：摘要只能被覆盖区间严格过去的 token 读取（cover_end 显式记账）。任何注意力路径改动必须通过泄漏测试：打乱输入末尾 10%，前 90% logits 必须 0 diff。

## 复杂度与标度

- 每 token 前向 **~66d² ≈ 4.3M MACs**（d=256, b=16, T=2, m=64），对 n 近似常数；对 full-d2 的成本交叉点 n≈2.5k
- read 与显存严格线性；选择打分 (n/b)²：65k 占 1.5%，1M 占 ~20%，**~4.3M 与线性部分打平**（n ≈ 66·b²·d）。总量 O(n + n²/b²)，≤100k 区间工程线性成立；4M 外由层级选择（顶层预筛）恢复近线性
- **KV cache** = level-0 状态 T·n·d + 摘要 n·d/(b-1) ≈ T·n·d，与 T 层 transformer 同阶；块填满后表示终态不变 → 天然流式可建（增量调度器为纯工程，未做）；生成 O(1)/token，金字塔同时是缓存与索引
- 参数与 n 无关（1.91M 从 512 用到 65k，零样本迁移）
- **短板**：小 n 常数打不过 FlashAttention（战场 ≥16k）；T=2 硬乘数；>4M 选择需层级化；共享权重容量税（~10 角色均摊表达力）

## 核心结果

| 实验 | 结果 | 意义 |
|---|---|---|
| 512 点火（dense / sparse） | 0.91–0.98 | 路由电路可训练 |
| 4096 冷启动（无课程，passkey） | ❌ 4000 步平坦 | 点火阈值现象（SNR 定式，见下） |
| 512→4096 课程迁移（passkey dense） | 零样本 0.9609 → 0.9922 | 电路尺度可迁移 ×8 |
| 512→4096 课程迁移（sparse m=64） | **0.9609** | **线性化不掉点** |
| sparse m=64 @16k 零样本 | **0.8984**（depth 全平 ≥0.885） | 反超 dense 0.7422；深位衰减治愈 |
| sparse m=64 @65k 零样本 | **0.9375** | 65k 可行（dense 无法运行） |
| **enwik8 char-LM @4096（halo, T=2）** | **bpc 1.5557**，posloss 平坦 | **反超参数匹配 full-d2 的 1.664** |
| MQAR 16对 @512（128 阶梯迁移） | **1.0000** | 阶梯点火通杀 |
| MQAR @4096（512 零样本迁移） | **1.0000（step 0）** | 迁移证据终极形态 |
| cycles_exact @4096 passkey | [0.031, 1.0] | 第二个 cycle 做几乎全部检索 |
| T 消融零样本 @4096 | T=1: 0.008 / T=2: 0.97 / T=4: 0.09 | T 须匹配训练；T=4 过冲毁解 |
| T=1 课程救援（passkey） | 0.9219 @4096 | 循环是容量放大器，不是魔法 |

基线对照：full @4096 passkey = 0.957；local = 0.000；local LM bpc ≈ 2.0。

## 机制与理论认识（论文素材）

1. **点火 SNR 定式**：冷启动失败 = 路由梯度信噪比不足，非架构能力。高原 = 边缘分布解的鞍部平面；深谷 = 检索电路盆地；逃逸速度 ∝ **SNR ≈ 监督密度 / 每 read 摘要数**（512→33 边缘，4096→257 阈下，128→9 弹道）。低于噪声地板时 SGD 退化为随机游走，点火呈顿悟式跳变。**配方：在看得见谷底的地方找到它，然后住在里面走遍所有尺度**。LM 冷启动成功且与课程打平（稠密监督天然高 SNR）——"长度 vs SNR"裁决 = SNR。
2. **零样本长度迁移**：共享算子与 n 无关，小尺度电路在大尺度零训练步工作（passkey 0.96@4096、0.90@16k；MQAR 1.0@4096）。**纪律**：迁移后常规 lr 续训会先拆电路（灾难性遗忘）——迁移评估一律取 `--eval_only` 零样本值。
3. **循环的证据与边界**：cycles_exact 证明迭代做工（passkey [0.03, 1.0]；MQAR 两 cycle 均衡 [0.95, 0.95]）；T=4 过冲 [0, .938, .375, 0] 证明算子在解处无不动点——"T 测试时旋钮"朴素形式死亡；T=1 可训练但弱一档，循环定位 = 容量放大器。
4. **边界解剖与 halo**：posloss 诊断显示残差全部长在块前 4 token（块内因果使块首局部上下文为零）；halo 平滑（前一块 ++ 本块 keys）一击消除，bpc 1.745 → 1.5557 并反超上界。诊断驱动的修复范例。
5. **截断洞察（用户）**：硬截分不是根本问题——模型只需识别截断即可经层级+迭代跨块重建联系。
6. **理论边界**：可证直径定理（任意两 token 依赖路径 ≤ 2T·log_b n 跳）；不可证收敛率（softmax 非线性）——以经验收敛曲线代替。

## 生产配方

- **点火→迁移**：小尺度点火（passkey 512 / MQAR 128，bs 64, lr 1e-3/5e-4）→ `--resume_weights_only` 到目标 n；评估用 `--eval_only` 零样本值
- **LM**：直接冷启动，`--sparse --halo`，bs 32, lr 1e-3
- 判读指标：eval_exact / bpc / depth_exact / cycles_exact / posloss_mod_b / gnorm

## 否决清单（全部有实验依据）

| 方案 | 判决 | 依据 |
|---|---|---|
| AMR 树下降选择 | ❌ | beam myopia：scorer 被自身选择绑架；ST/退火/dense 接手均救不回 |
| shifted blocking | ❌ 毒性 | 512 bisect 显著拖慢 |
| per-cycle 出口损失 | ❌ 毒性 | 同上；强迫单 cycle 可解与分工冲突 |
| posxattn（新/旧） | ❌ 驳回 | 用户决定；新实现有混叠嫌疑 |
| poolaux（BOW 辅助） | ❌ 无效 | 只教"写什么"，不建立"怎么读" |
| bf16 嫌疑 | ✅ 清白 | fp32 同样失败 |
| 不动点训练 | ⏸ 搁置 | 用户否决（T=4 过冲是其存在理由，留档） |

## 文献定位

| 路线 | 代表工作 | 层级？ | 循环共享权重？ |
|---|---|---|---|
| 层次/多尺度 | H-Transformer-1D、FMMformer、Hourglass、Swin | ✅ | ❌ 权重不同、单 cycle |
| 深度循环 | Huginn、Ouro、Mixture-of-Recursions | ❌ 固定分辨率 | ✅ 深度方向 |
| 序列循环/记忆 | Block-Recurrent、RMT、Infini-attention | ❌ | ✅ 沿序列 |
| 工业稀疏 | DeepSeek NSA、Kimi MoBA | 一层粗+细 | ❌ |

**层级 × 循环的交叉格是本工作的位置**；SparseRead 的选择机制与 MoBA/NSA 同族，差异在"摘要树上的块路由 + fine fanout + 跨尺度共享算子"。命名注意：H-Transformer-1D 已用过 "multigrid"，投稿建议 **V-cycle Transformer**。

## 下一步

1. **T=1 @LM**（T 乘数消融，最高性价比——若接近 T=2 则成本腰斩）
2. C3 kq=4 带宽、C2 m=257 稀疏税（摘要充分性最后未知数）
3. 16k LM 外推封顶图 + 30k 步预算
4. 论文写作（`paper/outline.md`，fig1-8 数据基本齐）
5. 远期：增量推理调度器（KV cache 落地）、chunked top-k、层级选择（>1M）

## 工程笔记

- 折叠批 CUDA grid y 上限（65535）→ `FOLD_CHUNK=8192`；CrossAttn 手工 matmul
- 远程 AutoDL（无外网）：`data/enwik8` 需上传（支持 zip，自动解压）；切分 90M/5M/5M
- `sweep.sh`（per-GPU worker）；日志 `runs/*.jsonl`；关键 ckpt：`sp512.pt`、`sp4096.pt`、`lm_mga_halo.pt`
- 词汇表分工：passkey/copying fillers=18-29；MQAR fillers=26-29、keys=10-25 无放回（曾因有放回矛盾标签冻结于 ln(10)，已修复）
