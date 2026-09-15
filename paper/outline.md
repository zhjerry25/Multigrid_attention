# 论文骨架（v0.3 工作稿，2026-09-14）

暂定题：**V-cycle Transformer: Trading Recurrent Depth for Attention Range**
（备选：Multigrid Attention Is All You Need —— 注意 H-Transformer-1D 已用过 multigrid 一词，正文引用并区分）

## 故事弧（四幕）

1. **问题**：全注意力 O(n²) 与"局部窗口/压缩记忆丢失精确检索"的两难。核心命题：**可以用网络深度与循环换注意力范围**——把完全图换成直径 2·log_b(n) 的树状层级图，迭代消息传递。
2. **方法**：V-cycle 块（平滑→限制→递归→延拓→后平滑，T 次循环，跨层跨 cycle 共享同一算子）+ SparseRead（全局块共享 top-m 摘要选择 + fine fanout）。因果 cover_end 记账。复杂度 O(T·n·(b+m)·d)。
3. **关键现象学（本文最独特的部分）**：
   - **点火阈值**：池化瓶颈处冷启动点火对比度 ∝ 1/摘要数；512 可点火、4096 不可（动力学失败，非能力失败）
   - **课程点火配方**：小尺度点火 → 大尺度零样本上岗；路由电路尺度可迁移
   - **循环的证据与边界**：cycles_exact 显示第二 cycle 做几乎全部检索；T=4 过冲（解处无不动点）——循环是容量放大器而非魔法
4. **结果**：线性化不掉点（4096, 0.9609 vs dense 0.97）；16k 反超 dense（0.8984 vs 0.7422，深位治愈）；65k 零样本 0.9375；**enwik8 char-LM @4096 bpc 1.5557 反超参数匹配 full-d2（1.664）**，halo 平滑消除边界饥渴（posloss 诊断驱动的修复案例）；标度四维度（FLOPs/显存/KV cache/生成）线性实证至 65k。

## 章节结构

1. Introduction（两难问题 + 贡献四条：架构、点火现象学与课程配方、零样本迁移、线性化不掉点 + 65k）
2. Method（V-cycle、共享算子、SparseRead、因果规则、复杂度分析）
3. The Ignition Problem（阈值模型 + 证据链 R2/R3/g + 课程作为解）
4. Experiments（合成套：passkey/MQAR/copying + T 消融 + m 前沿；真实套：enwik8 bpc + 外推；对照：full/local/MoBA 式强基线）
5. Analysis（cycles_exact、depth 指纹、T 过冲、截断鲁棒性）
6. Related Work（定位表见 topic.md；层级 H-Transformer-1D/FMMformer、循环 Huginn/MoR/Ouro、稀疏 NSA/MoBA、记忆 RMT/Infini）
7. Limitations（短 n 常数劣于 FlashAttention；LM 上界仍由 full 保持；收敛率为经验结果）

## 图表清单（映射到数据）

| 图 | 内容 | 数据来源 |
|---|---|---|
| fig1 | 架构示意（V-cycle + 层级树 + SparseRead） | 手绘 |
| fig2 | 点火阈值：4096 冷启动平 vs 512 曲线 vs 课程后 | runs/mga_T2_long, r2/r3, stage4096 |
| fig3 | 零样本迁移 ×8：512 ckpt 在 4096 逐步曲线 | runs/stage4096 |
| fig4 | T 消融 [1/2/4] + T=4 过冲 cycles_exact | 远程 eval_only 三连输出 |
| fig5 | m 前沿曲面（m×n） | Phase A1（sp4096.pt evals） |
| fig6 | 65k depth 桶 [1.0, 0.941, 0.857, 1.0] | S4 输出 |
| fig7 | enwik8 bpc 曲线（mga/full/local）+ 16k 外推柱 | Phase B runs/*.jsonl |
| fig8 | 循环必要性：MQAR 上 T=1 vs T=2 | Phase A2 |

## Related work 查撞车清单

- Awesome-Loop-Transformers（2026-09 仍更新，通读）
- 必引：H-Transformer-1D（层级+multigrid 词源）、FMMformer、Huginn、MoR、NSA、MoBA、Block-Recurrent、Infini-attention、RMT、DEQ（不动点搁置说明）

## 诚实声明（审稿人预期问题预答）

- "就是 H-Transformer-1D 绑了权重？" → 绑定+迭代是贡献本体，外推与 T 分析是它给不出的
- "短序列效率？" → 战场在 100k+；短 n 常数劣于 FlashAttention，以 FLOPs-性能前沿呈现
- "收敛率理论？" → softmax 非线性，证不了；给直径定理 + 经验收缩曲线
