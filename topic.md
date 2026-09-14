# Multigrid Attention：用循环层级结构交易注意力范围

**立项文档 · 2026-09-13**

## 一句话

把全注意力的完全图消息传递，替换为直径 2·log_b(n) 的树状层级图上的迭代消息传递；所有层级共享同一算子，整个 V-cycle 作为循环单元重复 T 次——**用循环深度换注意力范围**，换取对序列长度严格线性的复杂度、天然的长度外推性、和测试时可调的算力旋钮。

## 核心动机

- 全注意力 = 完全图上一步全连通，代价 O(n²d)。
- 本方案 = 局部计算 → 聚合成更大单元 → 再计算 → 再聚合，任何两个 token 经最近公共祖先（LCA）在 log 跳内必达。
- 每条边上的算子共享权重 = 循环；迭代多个 cycle = 迭代求解器，逐步逼近全注意力。
- 这是数值分析中多重网格（multigrid）V-cycle 的语义：局部平滑 → 限制到粗网格 → 粗层求解 → 延拓回细网格。

**冷水的部分**：这不是"凭空变出无限注意力"。聚合是有损压缩，远处 token 对之间的细粒度交互被压制。FMM 在物理问题中成立是因为相互作用核平滑衰减；语言依赖不平滑（一个代词指代五千词前的人名）。方案成立的全部赌注在于：**循环求精能否找回丢掉的细粒度交互**——这是可证伪的，是好事。

## 方法定义（MGA block）

输入 x ∈ R^{n×d}，块大小 b（如 32），共享算子 Θ（一个标准 transformer block）。一个 **V-cycle**：

1. **分块 + 平滑（smoothing）**：连续切成长度 b 的块，块内跑 Θ（局部注意力）。
2. **限制（restriction）**：每块用 learned cross-attention pooling 压出 k 个粗 token（k≥1，k 是带宽），得到 n/b 长度的序列。
3. **递归**：对粗序列用**同一个 Θ** 重复 1–2，直到顶层 O(1) 个全局节点。
4. **延拓（prolongation）**：细 token 以 query 身份 cross-attend 自己的父节点链，主动"读"全局信息。**必须是 read，不能是加法广播**——纯池化+广播在检索任务上必死。
5. **后平滑**：细层再跑一次 Θ，残差融合。
6. 整个 cycle 重复 **T 次**，权重全共享；可加 per-cycle 出口辅助 loss（deep supervision）。

### 复杂度（严格线性，无 log 因子）

第 ℓ 层有 n/b^ℓ 个节点，该层块内注意力代价为 (n/b^{ℓ+1})·b²d = n·b·d/b^ℓ，逐层几何衰减。求和：

$$\\sum_\\ell \\frac{n \\cdot b \\cdot d}{b^\\ell} = O(n \\cdot b \\cdot d)$$

**单 cycle 总代价 O(n·b·d)，总计 O(T·n·b·d)，对 n 严格线性。** 优于 H-Transformer-1D 的 O(n log n)，与线性注意力同阶——但记忆容量是 O(n/b) 的金字塔，而非 O(1) 固定状态。

## 关键设计决策

1. **延拓用 cross-attention read**（见上），这是检索能力的生命线。
2. **因果树**：生成式 LM 中粗 token 只能 summarize 自己覆盖的过去段（H-Transformer-1D 已证明层次注意力可做因果掩码）。
3. **跨尺度共享权重是假设，不是定理**：底层处理原始 token，顶层处理"摘要的摘要"，统计特性不同。赌注是语言的尺度自相似性（字→词→短语→句）。留消融旋钮：Θ 共享主体 + 每层极小 scale embedding / LoRA adapter。
4. **Shifted blocking**：每个 cycle 把分块边界错开 b/2（Swin shifted window 的零成本招），解决硬分块把天生该直连的 token 对隔开的问题。
5. **聚合用 learned cross-attention pooling，不用 mean-pool**；每块 k>1 个粗 token 作为带宽旋钮。
6. **位置编码**：块内相对位置 / NoPE + 粗层 block index。

## 文献定位：空的格子在哪

| 路线 | 代表工作 | 层级？ | 循环共享权重？ |
|---|---|---|---|
| 层次/多尺度注意力 | H-Transformer-1D (2021, 明确引 multigrid, O(n log n))、FMMformer (2021)、Hourglass、Swin | ✅ | ❌ 每层权重不同，单 cycle |
| 深度循环/权重共享 | Huginn (Geiping et al., NeurIPS 2025, 3.5B)、ByteDance Ouro、Mixture-of-Recursions (2025) | ❌ 固定分辨率 | ✅ 深度方向 |
| 序列方向循环/记忆 | Block-Recurrent Transformer (2022)、RMT、Infini-attention | ❌ | ✅ 沿序列 |
| 工业级稀疏注意力 | DeepSeek NSA、Kimi MoBA、Hierarchical Top-P Sparse Attention (2026) | 一层粗+细 | ❌ |

**层级 × 循环的交叉格是空的。** 注意：

- H-Transformer-1D 五年前已用过 "multigrid" 一词，命名建议 **V-cycle Transformer** 更安全。
- 审稿人必然质疑"这就是 H-Transformer-1D 绑权重跑几遍"。应对：承认这一点，把仗打到他们给不出的两张图上——**长度外推**和 **T-scaling**。
- LT2 (2026, 循环×线性混合器，引用待核实) 已抢占"循环×线性复杂度"叙事；差异化必须讲清：他们是无定形线性混合器，我们是结构化层级，卖点是外推性 + 记忆容量。
- 参考图谱：Awesome-Loop-Transformers（2026-09 仍在更新，写 related work 前通读查撞车）。

## 三个真正的卖点（比"无限注意力"更准确的 claim）

1. **长度无关的算子 → 天然外推**：块内注意力、聚合、修正，无一依赖 n。训练 8k，原则上直接跑 1M。Transformer-XL、MoR 都做不到。
2. **金字塔记忆 > 固定状态瓶颈**：Mamba/线性注意力把历史压进 O(1) 状态；本方案存 O(n/b) 个摘要向量，容量随长度增长，且可流式增量构建。
3. **T 是测试时算力旋钮**：同模型多循环几轮换更准的长程交互，接 test-time compute 叙事。

## 可证伪假设与最小实验（100M 参数级以内，按优先级）

- **H0 · needle / passkey（生死线，最先做）**：32k 上下文藏随机数。LM loss 收敛是平均-case，needle 是 worst-case——loss 收敛不代表检索能力回来。找不到 needle，整个方案退回"好的编码器、差的 LM"。
- **H2 · 长度外推（赢面最大的一张图）**：train 8k → test 64k/256k，对比 Transformer-XL、Mamba 的 loss 曲线。
- **H1 · 收敛性（理论故事）**：短序列上 V-cycle 的 loss 随 T 单调下降并逼近全注意力 loss。
- **H3 · 效率前沿**：固定 FLOPs 下 loss-vs-context-length 的 Pareto 前沿，对比 dense / sparse / 线性基线。

任务顺序：合成任务（passkey、远距离 associative recall、copying）→ PG-19 / 长文档 LM。

消融轴：T ∈ {1,2,4} × k（粗 token 带宽）∈ {1,4} × 跨尺度共享 vs 独立权重 × shifted blocking on/off。

## 风险清单

1. **池化信息瓶颈**：摘要容量 vs 覆盖 token 数。缓解：k>1 带宽 + cross-attention 聚合 + prolongation read 路径。
2. **检索保真度**：见 H0，这是唯一生死线。
3. **循环训练稳定性**：T cycles × log n 层 = 很深的有效深度。借鉴 Huginn 的归一化/初始化配方 + per-cycle 出口 loss。
4. **收敛率理论别当 theorem 卖**：经典 multigrid 证明依赖线性算子，softmax 非线性，大概率证不出。写成 open question + 经验收敛曲线。
5. **Kernel 工程**：层次 gather 不规则，用块对齐 mask / FlexAttention；常数因子决定短序列打不过 FlashAttention，战场选在 100k+。
6. **test-time compute 叙事拥挤**（Huginn / Ouro / MoR / Astra 传闻）：差异化锚定结构化层级独有的外推性与记忆容量。

## 决定性结果（2026-09-13 晚，远程 CUDA）

**H0 彻底通过，根因定论，课程配方确立。方法方向验证成功。**

### 判决表（passkey exact-match）

| 实验 | 配置 | 结果 |
|---|---|---|
| mga @4096 冷启动 | bf16 (R2) / fp32 (R3)，无任何开关 | ❌ 4000 步平坦 |
| full @4096 | bf16，lr 1e-3 | ✅ 0.957（同机对照） |
| mga @512 冷启动 | fp32 (g) / CUDA (R4) | ✅ 0.93–0.98 |
| **mga 512→4096 课程（R5）** | resume，零样本 | **✅ 0.9609（零 4096 训练步）→ 0.9922** |
| poolaux @4096 (R6) | BOW 辅助损失 | ❌ 无效 |
| exitloss / shift @512 | bisect 单开关 | ❌ 显著拖慢（毒性确认，已回退） |

### 根因定论

4096 冷启动失败 = **点火（ignition）失败，不是架构能力**。机制：串联信用赋值（pool 学"写什么" × read 学"找哪个"）需要 bootstrap，其每步梯度对比度 ∝ 1/（每 read 的摘要数）：512→33 个（阈值之上），4096→257 个（阈值之下）。排除项：bf16、lr、bs、树深（两尺度同为两级池化）、带宽（80bit≪256 维）、全部新组件、CUDA。

### 三个论文级发现

1. **零样本长度迁移 ×8（当前最漂亮的数据）**：512 顿悟 checkpoint resume 到 4096，**零 4096 训练步** eval exact 0.9609，终到 0.9922；depth 四分桶全部 ≥0.96（指纹：检索能力不随 needle 深度衰减）。共享算子的路由电路尺度可迁移——H2 外推的第一实证，且强于预期（连 finetune 都省）。
2. **循环在做工的直接证据**：R5 上 cycles_exact = [0.031, 1.0]——第二个 cycle 完成几乎全部检索。"迭代求精找回细粒度交互"从假说变为观测事实。
3. **课程点火配方**：小尺度点火 → 大尺度上岗。共享算子参数与 n 无关（RoPE buffer 非持久、跨 n 状态字典同构），课程迁移零架构成本。进入论文成为重要章节（用户决策）。

### 否决与回退

- **poolaux**：只教 pool"写什么"，不建立 read"怎么读"——4096 单考无效，开关保留默认 off。
- **exitloss**：强迫单 cycle 可解与两 cycle 分工冲突，512 上显著拖慢。已回退。
- **shift**：用户洞察成立——**硬截分不是根本问题，模型只需识别截断即可经层级+迭代跨块重建联系**；shifted blocking 不必要且有害（退化树 + 尾段孤岛）。已回退。
- 默认训练配置 = 裸机（无新开关）+ 课程点火。posxattn 新旧实现均未完全定性，默认 off。

### 新工具

`--eval_only`：加载 checkpoint 在任意 n、任意 T 下求值（T 只是前向展开次数、权重同构；n 同理）——一次训练，T 消融 × 长度外推全部零样本拿到。

## T 指纹 × 零样本外推 × T=1 救援（2026-09-13 深夜，全部零训练成本）

**T 消融（stage512.pt 零样本 @4096）**：

| T | eval exact | cycles_exact |
|---|---|---|
| 1 | 0.0078 | [0.0] |
| 2（训练值） | **0.9688** | [0.0, 0.938] |
| 4 | 0.0938 | [0.0, 0.938, 0.375, 0.0] |

- T=1 零样本失败：循环电路按训练时的 cycle 数分工，拆走第二个 cycle 即崩——预期内。
- **T=4 过冲（v0.1 最有科学价值的意外）**：cycle 2 已解出（0.938），cycle 3–4 把解**破坏掉**（0.375→0.0）。共享算子在解处**没有不动点**，它只是为恰好 2 次应用优化的。"T 测试时旋钮"朴素形式死亡（用户"T 固定为超参数"的决策被数据证实）；复活条件 = 训练时随机化 T，让解成为吸引不动点（T-dropout 回归，列为 v0.2 开放问题，接 DEQ 文献）。
- **T=1 可被课程救回（0.9219@4096）**：循环不是检索的必要条件，是有效容量放大器。用户定论："循环本来就不是魔法，只是稍微增加的等效容量。"原"T=1 弱于 T≥2 = 指纹"的预期被推翻，如实记录；循环优势预期随任务难度增长（MQAR/多跳/长 n 再验证）。

**零样本外推 ×32**：stage512.pt @16384：exact **0.7422**（digit 0.9453），depth = [0.946, 0.655, 0.577, 0.722]——优雅衰减：近位 needle 仍 0.95，深位降至 0.6–0.7。归因：read 的 key 数 ×4（1025）→ 路由对比度再稀释（同一根因的第三形态，指向 v0.2 治理方向）。对照需求：full@512→16k（RoPE 直接外推通常崩溃，若 full≈0 则此图 killer）。

posxattn（新实现）驳回（用户决定）；bisect 审计终结。

**v0.1 收尾**：方向验证完成。生产配方 = 裸机 + 课程点火（512 顿悟 → 目标 n 上岗）；实验 flag 全部默认关，保留为审计轨迹。

## v0.2 AMR 选择路径（已实现，`--amr`，2026-09-14）

**动机**：v0.1 的稠密 read 是 O(n²/b²)——16k 时深位 needle 衰减（0.58–0.72，路由对比度稀释）、65k 时显存不可行。AMR 把 level-0 read 换成**树下降选择 + 细粒度直连**，降到 **O(n·b·log n)**，验收硬标准 = 线性化且不掉点（用户标准）。

**机制**（`AMRRead`，level-0 专用，高层保持稠密）：
1. **beam 下降**：自顶层候选（cover_end ≤ i 的顶层摘要 + null 吸收态）逐层打分（q·摘要 softmax，权重求和 = coarse 上下文）→ argmax 选 1 个 → 其子节点成下一层候选。子节点平铺父区间 ⇒ 因果自动安全（仅顶层需掩码）。
2. **末层 top-m fanout**（m=4）：胜出的 level-1 摘要展开为其 level-0 原始 token（m·b 个），token 级 cross-attn 直连——"直接切出密码"的机制实现，治深位衰减的治本手段。
3. 每 token key 数 = b·log_b n + m·b + b（本地块）。65k 时 read 从 16.7G 对降到 ~3M 对。

**梯度**：softmax 权重与 fine 注意力可微；argmax/topk 索引 detach（MoBA/NSA 惯例）。scorer 与 fine reader 两个小 CrossAttn，跨层跨 cycle 共享（全共享哲学）。
**约束**：kq=1、无 shift、≥2 摘要层；与 dense ckpt 参数不兼容（须 AMR 从头点火 512）。
**已验证**：泄漏测试 amr 变体 0 diff。

**远程验收门**：512 点火 → 4096 resume ≥0.96（不掉点主门）→ 16k 零样本 ≥0.9 且深位桶提升 → 65k 前向可跑 → MQAR 泛化。

## v0.2 方向（从 v0.1 证据生长出来，按优先级）

1. ~~不动点训练~~（用户否决，搁置）。
2. **AMR 选择路径（主线，已实现待验收）**。
3. **循环价值的压力测试**：MQAR / 多跳 / 更长 n 上 T=1 vs T=2，找到循环真正必要的任务区间（passkey 上已证明非必需）。
4. **full@512→16k 对照训练**：外推赢面的定量证据。
5. **真实语料**：enwik8/PG-19 + 课程配方，LM loss 主指标；fixed-FLOPs Pareto 前沿（H3）。

## 决策记录（2026-09-13，第二轮讨论）

- **H0@512 通过**：mga T=2 eval exact **0.9336**（full 0.98 / local 0.00，判别结构完全符合预期）。lr=1e-3 + bs=64 是关键变量，架构无罪。
- **主瓶颈 = 监督带宽，不是位置/带宽**：鸡生蛋串联信用赋值（pool 学条件路由难 × read 学选择易）+ 每序列仅 5 个 loss 位 + 共享权重跨角色梯度干扰。posxattn A/B：有增益但非主因（step 3000：0.90 vs 0.84）。
- **T 固定为训练超参数**（用户决策）：主消融 T∈{1,2,4} 各自独立训练；T-dropout 搁置；"测试时调 T" 只留一个探索性小实验（train T=2 → eval T∈{1,4}），不承担主 claim。
- **"注意力不平滑"定论**：multigrid 对本工作只是图结构蓝图（直径定理可证），平滑性分析不成立也不追求——语言依赖本来就是尖峰的，要的是路由不是扩散。
- **read 成本诚实账**：v0 的 read 是 O(n²/b²)，**不是线性**。100k+ 声明前必须换线性变体（root-path 或 top-k 摘要选择）。

## v0.1 设计清单（已全部实现 ✅，2026-09-13 第三轮）

1. **per-cycle 出口损失** ✅（`--exitloss`，w_t 递增 0.3→1.0；eval 自动记录 cycles_exact = H1 收敛曲线）。
2. **长度课程** ✅（`--curr 512,1024,2048,4096`，前半训练变长采样；纯数据侧，共享权重天然支持）。
3. **Pool/Read 位置编码** ✅（`--posxattn`；Read 的 q 在 token index、k 在 cover_end 旋转——q·k 直接编码精确距离）。
4. **kq>1 带宽** ✅（`--k`，要求 kq | b；Read 掩码按 cover_end 泛化）。
5. **shifted blocking** ✅（`--shift`，奇数 cycle 偏移 b/2；head/mid/tail 分段保证部分块因果干净；泄漏测试覆盖）。
6. **每角色 LayerNorm**（`--roleln`）：未做，训练不稳时再加。
7. **线性 read 变体**：未做，>16k 实验的前置条件；当前 read 为 O(n²/b²)（诚实账）。

评测套（`mga/data.py`）：passkey（含 depth_exact 深度分桶 = needle×T 指纹图数据）、copying、**MQAR**（16 对 key→value 干扰中的内容寻址，注意：filler 词表改为 18-29，10-17 保留为 MQAR keys，与旧 512 运行不严格可比）。
基线：full（`is_causal` flash 路径，4096 在真卡可跑）、local（感受野=窗口，深度不变）。
训练配置（真卡）：bs 256 + lr 2e-3 + steps 4000 + bf16 + EMA 0.999 + ckpt 周期保存。
多卡：`sweep.sh`——每配置独占一卡、自动排队，12 个 job（T×kq 主矩阵 + full/local + mqar/copying 泛化），`bash sweep.sh` 一键。

## v0.2 旗舰机制：AMR（自适应网格加密）

用户直觉（"切分本身应该可学习，直接切出密码"）的正确形态。文献 verdict：端到端学硬切分历史记录差（离散边界不可微，ST/Gumbel 方差大）；可行的是"用注意力自身信号做选择"（NSA 选择分支、MoD 路由）。机制：**cycle t 的 read 注意力权重 = 免费的误差估计器**，注意力强的摘要块在 cycle t+1 被展开为细 token 直连可读。可微、无需额外模型、与 query 条件化池化是同一思想。命名为 AMR-for-attention，直接锚定数值分析文献。

## 权重共享的合理性（第二轮结论）

全共享保留为**默认**，因为它是两个 headline claim 的承重墙：跨 cycle 共享 = 不动点/迭代求解叙事的前提（权重不同就没有不动点可言）；跨尺度共享 = 长度外推的前提（n 翻倍时树长出新层，只有共享算子能处理训练中没见过的深度）。代价是跨角色梯度干扰（Θ 在 T=2 下以 ~10 种角色被调用）。对冲不是拆分权重，而是加 roleln / per-level 小 adapter 做消融：**若全共享 ≈ 加 adapter，则"语言尺度自相似、算子可复用"本身是可发表的发现**。

实验报告原则：x 轴用 FLOPs/墙钟而非步数；对比需含**匹配预算的强稀疏基线**（只打 full 会被挑）。

## 理论边界（能证 / 不能证）

- **证不了**：V-cycle 对全注意力不动点的收敛率。经典 multigrid 证明靠线性迭代矩阵 + 光谱平滑性，softmax 注意力是输入依赖非线性算子，两条腿全断。写成 open question + 经验曲线，不写 theorem。
- **能证**：
  - 直径定理：单 cycle 后任意两 token 依赖路径 ≤ 2·⌈log_b n⌉ 跳，T cycles 后 ≤ 2T·log_b n（"深度换注意力"的严格版本，一页内证完）。
  - 复杂度上界 O(T·n·b·d)（root-path read 变体，严格线性）。
  - 表达力 ⊇ 全注意力（k=b 时 restriction 是双射；注记级别，别当主定理）。
- **经验理论四张图**：loss-vs-T 收敛曲线；表示收缩率；训练后注意力矩阵数值秩（接 Linformer 那支，作为假设+验证）；needle 成功率 vs 距离 × T——T=1 应随距离衰减、T≥2 应平，这是 log 跳路由的指纹图，理论与实验咬合最紧的一张。
