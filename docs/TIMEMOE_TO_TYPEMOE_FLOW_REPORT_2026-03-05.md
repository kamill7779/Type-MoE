# Time-MoE 原始流程与当前改造流程说明（详细版）

更新时间：2026-03-05  
适用代码仓库：`C:\Users\kamil.liu\source\Type-MoE`

---

## 1. 这份报告讲什么

这份文档把两件事讲清楚：

1. **Time-MoE 原本是怎么跑的**（从数据到预测，路由怎么打分、怎么选专家、怎么融合）。  
2. **我们现在改成了什么**（类型约束路由 + 异构专家 + 配置驱动），以及这套改造在工程上是怎么落地的。

文风尽量直白，不用论文腔，但会尽可能细，方便直接抽成论文素材。

---

## 2. Time-MoE 原本流程（改造前的标准逻辑）

## 2.1 数据进入模型前

原始训练入口会把时间序列切成固定窗口（见 `time_moe/datasets/time_moe_window_dataset.py`）：

1. 输入 `input_ids`：窗口前 `L` 个时间点。  
2. 标签 `labels`：窗口后移一位的序列（自回归目标）。  
3. 可选 `loss_masks`：对 padding 或无效位置做掩码。

所以训练目标本质上是：  
“给定历史序列，预测后续 token（或 horizon）。”

## 2.2 初始 embedding

Time-MoE 不是单线性 embedding，而是门控 embedding（`TimeMoeInputEmbedding`）：

`emb = act(Wg x) * (We x)`

解释：

1. `We x` 给出基础隐表示。  
2. `Wg x` 经过激活后给出逐维门控强度。  
3. 两者逐元素乘，得到初始 token hidden。  

形状上是：`[B, L, C_in] -> [B, L, H]`。

## 2.3 主干层结构（逐层迭代）

每个 `TimeMoeDecoderLayer` 结构固定：

1. `RMSNorm`  
2. `Self-Attention`（带因果约束）  
3. 残差相加  
4. `RMSNorm`  
5. `FFN`（dense MLP 或 MoE 专家层）  
6. 残差相加

这一层会堆叠 `num_hidden_layers` 次。  
注意：**跨时间步信息主要由 self-attention 提供**，FFN/MoE 主要是逐 token 的特征变换。

## 2.4 原始 MoE 路由打分（标准 top-k）

原始 `TimeMoeSparseExpertsLayer` 做法是：

1. 把 hidden 展平：`[B, L, H] -> [B*L, H]`。  
2. 线性 gate 打分：`router_logits = gate(x)`，得到 `[B*L, N_experts]`。  
3. `softmax` 得到专家概率。  
4. 直接做 `top-k`（例如 k=2）。  
5. 只把 token 分发给选中的专家（稀疏执行）。

也就是说，**评分由 gate 决定，不是专家自己评分**。

## 2.5 原始专家执行与融合

对每个专家：

1. 找到分配给该专家的 token 索引。  
2. 只跑这部分 token。  
3. 输出乘以对应路由权重。  
4. 用 `index_add_` 回填到总输出。

此外还有一个 **shared expert** 常驻分支：

1. 所有 token 都经过 shared expert。  
2. 再乘 `sigmoid(shared_expert_gate)`。  
3. 与稀疏专家输出相加。

## 2.6 原始 aux loss（负载均衡）

原始实现沿用 Switch 风格负载均衡损失，大意是：

1. 统计每个专家被选中的频率（`f_i`）。  
2. 统计每个专家平均路由概率（`P_i`）。  
3. 最终 `aux ~ num_experts * sum(f_i * P_i)`。

目的是避免所有 token 都挤到极少数专家。

## 2.7 预测头与训练目标

`TimeMoeForPrediction` 里：

1. 多个 horizon 头（`horizon_lengths`）可并行输出。  
2. 主损失是 AR loss（默认 Huber），对 `[B, L, horizon, C]` 做对齐。  
3. 若开启 aux，则把路由辅助损失加到总 loss。

推理时按所需 horizon 取对应输出头。

---

## 3. 我们现在的改造流程（Type-MoE 方向）

这一节对应当前仓库已改动代码。

## 3.1 核心改造目标

这次不是只换几个专家名字，而是改了三层逻辑：

1. **路由策略升级**：从标准 top-k 改成 typed top-k（类型约束）。  
2. **专家体系升级**：从同构 MLP 变成可插拔异构专家。  
3. **配置入口升级**：支持 YAML/JSON 覆盖模型配置，便于实验编排。

## 3.2 Typed 路由（类内预选 + top-k）

新流程（在 `modeling_time_moe.py`）：

1. `router_logits -> softmax` 得到全专家概率。  
2. 若 `router_mode = typed_topk`：  
   每个类型（trend/cycle/anomaly）只保留该类型内分数最高的专家，其余清零。  
3. 在“类型冠军”集合上做 `top-k`。  
4. 得到最终 `selected_experts + routing_weights`。

这等价于：  
**“先每类选第一，再跨类做 top-k”。**

## 3.3 actual_k 与辅助损失对齐

现在实现了 `actual_k`：

1. 不再盲目使用配置里的 `top_k`。  
2. 会根据当前 token 有效候选数做 clamp。  

同时 aux 计算也改成按 typed 过滤后的概率和实际 top-k 统计，避免“训练目标和真实路由行为不一致”。

另外新增了一个可选的 `type_diversity_loss`，用于抑制类型塌缩。

## 3.4 专家插件化（Registry）

新增 `time_moe/models/experts/` 目录，提供：

1. `BaseTokenExpert`：统一接口基类。  
2. `registry.py`：专家注册与构建。  
3. `build_expert(...)`：按配置动态实例化专家。

这样专家不再写死在 `ModuleList([TimeMoeTemporalBlock...])`，而是由 `custom_expert_specs` 驱动。

## 3.5 flat 专家与 seq 专家并存

现在支持两种专家接口：

1. `flat`：输入 `[M, H]`，适合纯 token 稀疏执行。  
2. `seq`：输入 `[B, L, H]`，适合需要完整序列上下文的算子（FFT、分解、序列注意力）。

为了不破坏语义，当前对 seq 专家采用 **S1 策略**：

1. seq 专家先在完整 `[B, L, H]` 上计算。  
2. 展平后只取被路由选中的 token 位置参与聚合。  
3. 未被选中的位置结果直接丢弃。

这保证了算子正确性，但会带来额外计算开销（这是已知权衡）。

## 3.6 已接入的异构专家（当前代码）

当前仓库已加入 4 类专家实现（token-expert 版本）：

1. `NBeatsTokenExpert`（trend, flat）  
2. `AutoformerTrendExpert`（trend, seq）  
3. `AutoformerCycleExpert`（cycle, seq）  
4. `FedFormerCycleExpert`（cycle, seq）  
5. `AnomalyTokenExpert`（anomaly, seq）  
6. `MLPTemporalBlockExpert`（baseline, flat）

对应外部参考源码仓库（已拉到 `thirdparty/`）：

1. Autoformer  
2. FEDformer  
3. Anomaly-Transformer  
4. N-BEATS

注意：这里是“算子迁移 + 接口适配”，不是把整套外部模型原封不动塞进来。

## 3.7 配置驱动入口

新增配置字段（`TimeMoeConfig`）包括：

1. `router_mode`  
2. `expert_types`  
3. `expert_type_map`  
4. `norm_topk_prob`  
5. `jitter_noise`  
6. `type_diversity_factor`  
7. `custom_expert_specs`  
8. `freeze_strategy` 等

并新增了训练入口参数：

1. `--model_config_override`（YAML/JSON）  

配套示例配置：`configs/typed_experts/base.yaml`。

---

## 4. 两套流程并排对比（可直接放论文方法章节）

## 4.1 路由决策

原始 Time-MoE：

1. `softmax -> top-k`（全专家直接竞争）

当前改造：

1. `softmax -> 类型内预选 -> top-k`（先保证类型覆盖，再竞争）

## 4.2 专家形态

原始 Time-MoE：

1. 专家基本同构 MLP block。

当前改造：

1. 专家异构化（MLP/N-BEATS/Autoformer/FEDformer/Anomaly）。  
2. 支持 flat + seq 两种接口。

## 4.3 稀疏执行

原始 Time-MoE：

1. 纯 token 稀疏 dispatch。

当前改造：

1. flat 专家继续 token 稀疏。  
2. seq 专家采用 S1（全序列算子 + 选中 token 聚合）。

## 4.4 辅助损失

原始 Time-MoE：

1. 标准负载均衡 aux。

当前改造：

1. typed-aware aux（与 typed 路由行为对齐）。  
2. 可选 type diversity 正则。

---

## 5. 从“数据到预测”的改造后全流程（一步一步）

1. 数据窗口进入模型（`input_ids`、`labels`、`loss_masks`）。  
2. 门控 embedding 生成初始 token hidden。  
3. 进入第 1 层 decoder：  
   self-attn 融合上下文 -> typed MoE 路由 -> 专家融合 -> 残差。  
4. 重复到第 N 层 decoder。  
5. final RMSNorm。  
6. horizon 输出头给出预测。  
7. 训练时：`AR loss + router aux (+ diversity)`。  
8. 反向传播同时更新：  
   embedding / attention / gate / experts / output heads。

---

## 6. 你写论文时可以强调的点

## 6.1 方法贡献角度

1. 在 Time-MoE token 路由框架内引入“类型约束路由”，增强可解释性。  
2. 在不抛弃原始架构的前提下，将专家扩展为异构算子池。  
3. 给出 seq 专家与 token 路由冲突下的可执行工程方案（S1）。

## 6.2 工程可复现角度

1. 配置驱动（`custom_expert_specs`）可复现实验编排。  
2. thirdparty 源码镜像可追溯专家算子来源。  
3. 保留 shared expert，避免基础表示能力断层。

## 6.3 风险与诚实表述（建议在论文里写）

1. seq 专家在 S1 下会增加计算开销。  
2. 类型先验划分（trend/cycle/anomaly）可能与数据集偏好不完全一致。  
3. typed 约束需要配合 aux/diversity，否则可能出现类型塌缩或组内垄断。

---

## 7. 当前实现状态说明（便于你引用）

截至本报告版本：

1. Typed routing、expert registry、异构专家类、配置覆盖入口均已落地到代码。  
2. 已完成分阶段代码提交。  
3. 尚未进行完整训练评测闭环（你当前要求是先盲写落地）。

---

## 8. 一句话总结

原始 Time-MoE 是“**统一专家池 + 全局 top-k**”；  
当前改造后的 Type-MoE 是“**类型约束路由 + 异构专家池 + 配置驱动集成**”。

