# 论文路线图与实验集合（map）

**版本**：v0.5-frozen（2026-09-18）。架构已冻结，本文档是论文工作的唯一指挥图：claim 驱动、实验闭环、严谨性纪律。**讨论与修改都先改这里，再动实验。**

## 0. 冻结架构（唯一被测对象）

V-cycle（halo 平滑 → 池化限制 → 递归 → 延拓 → 后平滑，T=2 循环）+ SparseRead（块共享 top-m=64 摘要选择 + top-mf=4 原始块直连）+ kq=1 + 全共享权重。d=256, b=16。1.91M 参数与 n 无关。**任何新实验必须用冻结配置**；新想法先记到 §10  parking lot。

## 1. 论文定位与 claims

题目方向：*V-cycle Transformer: Trading Recurrent Depth for Attention Range*。

| # | Claim | 支撑实验 | 状态 |
|---|---|---|---|
| C1 | 层级×循环架构可做**精确长程检索**（不是只有摘要语义） | passkey/MQAR 系统套件（§4） | ✅ 初证 |
| C2 | **线性化不掉点**（等参数打平/反超全注意力） | LM 主表（§2 E1）+ 检索（§4） | ✅ 初证（4096） |
| C3 | **零样本长度迁移**：一次训练，多尺度上岗 | 迁移曲线（§2 E2、§4 阶梯） | ✅ 初证（×8/×32/满分） |
| C4 | **标度优势**：FLOPs/显存/KV cache/生成成本 对 n 线性或常数 | 性能分析（§5）+ 推理示范（§6） | ✅ 理论+65k 实证 |
| C5 | **循环 = 等效容量乘数**（零参数增长换质量） | T 消融 + LM T1/T2（§7 A1） | ✅ 初证（0.24 bpc） |

## 2. 主结果实验（Table 1/2 主图）

### E1 · LM 主表（enwik8 char-LM, n=4096）
冻结配置 vs 基线组（§3）。30k 步收敛版，**3 seeds**（mean±std）。
- 当前：mga 1.5557 / full-d2 1.664 / full-d8 1.243 / local ~2.0（单 seed）
- 目标：mga ≤ full-d2 且 tok/s 优势随 n 展开（E2 兼）

### E2 · 外推曲线（核心图）
bpc vs n ∈ {4096, 16384, 65536}：mga 零样本（E1 ckpt + `--eval_only`）vs full-d2 直接外推 vs local。预期 mga 平缓、full 崩。65536 处 full 报 OOM 也是数据。

### E3 · PG-19 子集（第二语料，书籍长文档）
byte-level（免 tokenizer、免离线包问题）。**子集约 1-2GB**（前 ~200 本书），90/5/5 切分，本地准备后上传。配置同 E1。若 PG-19 上传困难 → The Pile 切片备选。
- 可选加强：TinyStories/Pile 切片的 100M 级"预训练示范"（loss 曲线 + 样本生成展示）

## 3. 基线矩阵（强方法，全部自实现——远程无外网）

| 基线 | 实现 | 风险 |
|---|---|---|
| full（dense flash） | 已有 | 无 |
| local（block-local） | 已有 | 无 |
| **DeltaNet-lite**（线性注意力/SSM 代表） | 纯 PyTorch chunked DeltaNet（~100 行）；先在小任务对 full 做 sanity | **中**：kernel-free 实现的数值/速度；论文标注 "lite" |
| **Segment-memory（Transformer-XL 式）** | 基于现有 Block：段内注意力 + 上一段状态缓存（~60 行） | 低 |

备注：mamba_ssm 官方实现需要 CUDA kernel 编译（离线机不可行），故用纯 torch 线性注意力代表 SSM 家族，并在文中如实说明实现差异与规模限制。

## 4. 检索压力测试（系统性，判据 = ≥99% solve）

### 4a · passkey 系统阶梯
n ∈ {512, 1k, 2k, 4k, 8k, 16k, 32k, 65k}，512 点火后逐级 `--resume_weights_only`；**3 seeds**；报 exact（mean±std）+ depth_exact 四桶。同时出**迁移曲线**（每级零样本值 vs 续训值）。

### 4b · MQAR 压力分析
- n_pairs ∈ {16, 32, 64} × n ∈ {512, 4096}，阶梯点火
- T=1 vs T=2（循环必要性的硬任务版）
- 多 needle passkey 变体（2/4 个 needle 同序列，顺带回答"单目标检索是否是巧合"）

### 4c · 故障分析
对所有 <99% 的点，报 depth 分桶 + cycles_exact，给出失败形态（定位错/绑定错/精度漂）。

## 5. 标度与性能分析（理论 + 实测，无 kernel 优化）

- **理论**：每 token FLOPs 公式表（已核定：~66d² + n·d/b²）；交叉点 ~2.5k（vs full-d2）/~4.3M（选择项）
- **实测**：tok/s vs n（日志已有）；**峰值显存 vs n**（给 `--eval_only` 加 `torch.cuda.max_memory_allocated` 日志，2 行）——mga 平缓 vs full 爆炸的显存曲线是 §C4 的直接证据
- 表格：n ∈ {4k, 16k, 65k} × {FLOPs, 显存, tok/s} × {mga, full, local, linear}

## 6. KV cache 推理示范（工程项目，~300 行）

**增量 V-cycle 引擎**（`mga/infer.py`）：块填充状态机 + 缓存管理（level-0 状态 T·n·d + 金字塔摘要）。
- **等价性测试**（泄漏纪律的延伸）：增量前向与整段前向逐位置 logits 全等（atol 1e-4），不过不收工
- **产出图**：生成每 token 延迟 vs 上下文长度（mga 平线 vs full 斜线）+ 65k needle 生成演示
- 这是 C4 在系统侧的落点，也是 demo 视频素材

## 7. 消融表（Table 3，冻结配置动一刀）

| 轴 | 取值 | 已有数据 |
|---|---|---|
| A1 循环数 T | 1 / 2 / 4（各训练） | T1: 1.795 / T2: 1.556（LM）；passkey 指纹已存 |
| A2 读取 m | 16 / 32 / 64 / 128 | passkey 前沿已存；补 LM 版 |
| A3 带宽 kq | 1 / 4 | 1.556 / 1.581（无税）✅ |
| A4 halo | off / on | 1.745 / 1.556 ✅ |
| A5 read | sparse m=64 / dense(m=257) | 1.556 / 1.544（微税 0.012）✅ |
| A6 权重共享 | shared / unshared | 待跑（容量税终审，LM） |
| A7 位置/exitloss/shift 等 | — | 否决清单引用 topic.md，不进正文 |

## 8. 严谨性纪律

1. **主表 3 seeds**（mean±std）；探索性图可单 seed
2. 检索任务判据统一 **≥99% solve**（exact-match），不足者进故障分析（§4c）
3. 所有对比同 token 量、同优化器配方；FLOPs 与 tok/s 同时报（x 轴双轨）
4. 基线自实现均须通过 sanity（同任务上对 full 的已知结果复核 + 泄漏测试）
5. 数据切分固定（enwik8 90/5/5；PG-19 子集同比例），种子固定，jsonl 全量归档

## 9. 优先级与时间线

| 优先级 | 内容 | 预计 |
|---|---|---|
| **P0** | 基线实现（DeltaNet-lite、XL-lite）+ sanity | 1-2 天（我写） |
| **P0** | E1 LM 主表 3 seeds + 4a/4b 检索套件 | 远程 2-3 天 |
| **P1** | E2 外推曲线 + §5 性能表（显存日志先加） | 1 天 |
| **P1** | §6 KV cache 引擎 + 延迟图 | 2 天（我写） |
| **P2** | E3 PG-19 子集 + 消融表 | 2 天 |
| **P2** | 初稿撰写（outline.md 已有骨架） | 持续 |

## 10. 风险与备选 / Parking lot

- **DeltaNet-lite 复现风险**：若与文献差距过大，降级为"线性注意力（gated）"并如实标注；或引用其公开结果做定性对比
- **PG-19 离线传输**：子集切片 → 失败则 Pile 切片 → 再失败则只保 enwik8 + 合成
- **KV 引擎 bug 风险**：等价性测试是硬门；不过就缩小 demo 到 passkey 生成
- **外推若在 65k 崩**：报告到 16k/32k 为止并分析失效形态，不遮掩
- Parking lot（不做但记录）：不动点训练（复活 T 旋钮）、AMR 树下降（已否决）、层级选择（>1M）、PG-19 全量、word-level tokenizer 对比

## 11. 论文版面设计（2026-09-18）

**版面原则：最强的证据放最前面，我们独有别人没有的能力做杀手图。**

### 正文

| 位置 | 内容 | 数据 |
|---|---|---|
| Fig 1 | 架构图（V-cycle + 层级树 + SparseRead） | 手绘 |
| **Fig 2（杀手图）** | **零样本迁移阶梯**：单个 ckpt × n ∈ {128…65k} 的 exact 曲线；full/local/linear 对照组在此图**无法出现**（无迁移能力=不存在曲线）——缺席本身就是证据 | stress 阶梯（MQAR 128→8192 零样本 ~1.0 已实证；passkey 同法） |
| Fig 3 | LM 主表 + bpc-vs-n 外推曲线 | enwik8（已有）+ PG-19 子集 |
| Fig 4 | 循环分析：cycles_exact 分担 + T∈{1,2,4} 容量-bpc 曲线 | LM T1/T2 已存；T=4 训练待跑 |
| Fig 5 | 标度与性能：FLOPs/显存/tok-s/生成延迟 vs n（mga 平线 vs full 爆炸） | 公式表 + 日志 + KV 引擎 demo |
| Table 2 | 消融（A1-A6） | 阶段 3 |

### 附录

点火现象学（SNR 证据链 R2/R3/g）、训练细节与超参表、否决方案简记（AMR/shift/exitloss/poolaux/posxattn）、预训练 demo（Pile/TinyStories 切片）、stress 阶梯全量表。

### 基线安置逻辑（哪里证明我们强）

- **检索任务（Fig 2 + §4）**：full = 质量上界（我们以线性成本打平）；local/linear/XL = 反例（随 n 崩或无法迁移）；**对照组在迁移图上的缺席即我们的胜利**
- **LM（Fig 3）**：full-d8 = 质量天花板（诚实不打，说明定位）；full-d2 = 参数匹配对手（我们反超 1.556 vs 1.664）；**赢面在 bpc-vs-n 与 bpc-vs-FLOPs 前沿**，不在单点
- **系统（Fig 5）**：full 的生成成本随 n 线性爬，我们平线——KV cache demo 的直接视觉证据
