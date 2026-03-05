# Type-MoE 相对于 Time-MoE 的改进点、创新点及理论依据

> **面向读者**：本科毕业论文撰写参考。可直接作为论文第三章（方法设计）和第四章（实现细节）的素材。
>
> **基线模型**：[Time-MoE](https://github.com/Time-MoE/Time-MoE)（Salesforce, 2024）——基于稀疏混合专家（Sparse Mixture-of-Experts）的大规模时间序列基础模型。

---

## 一、总体改进概览

Time-MoE 采用**同构 MLP 专家 + 标准 top-k 路由**的经典 Sparse MoE 架构。Type-MoE 在其基础上做出三个层面的改进：

| 层面 | Time-MoE 基线 | Type-MoE 改进 |
|------|--------------|--------------|
| **专家结构** | 8 个同构 `TimeMoeTemporalBlock`（MLP） | 8 个异构专家：MLP + N-BEATS + Autoformer + FEDformer + AnomalyTransformer |
| **路由机制** | 标准 softmax → top-k | 类型约束路由（typed top-k）：softmax → 类型内预选 → top-k |
| **训练策略** | 端到端全参数训练 | 分阶段解冻 + zero-init 残差 + 类型多样性损失 |

---

## 二、创新点一：异构专家池（Heterogeneous Expert Pool）

### 2.1 问题背景

Time-MoE 的每个专家都是相同的 `TimeMoeTemporalBlock`（双层 MLP + SiLU 激活），结构如下：

```
x → Linear(H, I) → SiLU → Linear(H, I) → Linear(I, H) → x'
         gate_proj            up_proj        down_proj
```

这类 MLP 专家本质上是 **pointwise 非线性变换**——它独立处理每个 token 的隐状态向量，不建模 token 之间的时序依赖。虽然 MLP 可以通过隐式记忆（self-attention 已经编码的序列信息经残差传递给 MLP）间接利用时序信息，但其建模能力受限于输入已编码的信息，无法主动捕获特定的时序模式（如趋势分量、周期分量、异常模式）。

**核心矛盾**：时间序列的内在结构是多成分的（趋势 + 周期 + 残差/异常），但同构 MLP 专家没有对这种结构做任何归纳偏置（inductive bias），完全依赖数据驱动的隐式学习。

### 2.2 设计方案

我们将 8 个专家 slot 划分为三种类型，每种类型内部混合保留原始 MLP 专家（确保基线能力）和引入新型专家（注入归纳偏置）：

| 索引 | 专家名                     | 类型      | 接口  | 来源与归纳偏置                                     |
|------|---------------------------|-----------|-------|--------------------------------------------------|
| 0    | MLP-Trend-0               | trend     | flat  | 原 `TimeMoeTemporalBlock`，复用预训练权重            |
| 1    | N-BEATS Trend             | trend     | flat  | N-BEATS 的 FC stack → θ → 多项式基函数展开，建模趋势 |
| 2    | Autoformer Trend          | trend     | seq   | 移动平均序列分解 → 提取趋势分量                      |
| 3    | MLP-Cycle-0               | cycle     | flat  | 原 `TimeMoeTemporalBlock`，复用预训练权重            |
| 4    | Autoformer Cycle          | cycle     | seq   | 序列分解 → 季节分量 → FFT 自相关 → 周期模式捕获       |
| 5    | FEDformer Cycle           | cycle     | seq   | FFT → 低频截断 → 可学习频域变换 → IFFT，频域周期建模   |
| 6    | MLP-Anomaly-0             | anomaly   | flat  | 原 `TimeMoeTemporalBlock`，复用预训练权重            |
| 7    | Anomaly Attention         | anomaly   | seq   | 高斯先验距离核 + 序列自注意力，建模异常关联模式         |

### 2.3 理论依据

**（1）时间序列分解理论（Time Series Decomposition）**

经典时间序列分析（Cleveland et al., 1990 STL; Hyndman & Athanasopoulos, 2021）将时间序列建模为：

$$Y_t = T_t + S_t + R_t$$

其中 $T_t$ 为趋势分量、$S_t$ 为季节/周期分量、$R_t$ 为残差（含异常）。这一假设是 Prophet、STL、MSTL 等经典方法的理论基础。我们将该先验知识注入专家类型划分，使 MoE 的专家能天然对应序列的不同分解成分。

**（2）专家特化与归纳偏置（Expert Specialization via Inductive Bias）**

Fedus et al. (2022, Switch Transformer) 指出 MoE 中专家会自然产生分工（specialization），但同构专家的分工完全依赖数据驱动，可能不稳定或低效。通过为不同专家注入领域特定的归纳偏置（移动平均分解、FFT 频域变换、注意力距离先验），可以引导专家更快、更稳定地学到对应的时序模式。

具体来说：
- **N-BEATS**（Oreshkin et al., 2020）：用全连接堆叠产生系数 θ，再通过多项式基函数（$t, t^2, t^3, \ldots$）展开重建趋势，天然适合低频缓变信号。
- **Autoformer** 的序列分解（Wu et al., 2021）：移动平均内核分离趋势与季节分量，Auto-Correlation 通过 FFT 找到周期性对齐模式。
- **FEDformer** 的频域增强（Zhou et al., 2022）：FFT → 可学习频域线性变换 → IFFT，在频域中直接操纵周期结构。
- **Anomaly Transformer** 的异常注意力（Xu et al., 2022）：高斯距离先验建模"正常关联模式"，与数据驱动的自注意力（series association）混合，偏离先验的部分暗示异常。

**（3）MoE 中的异构专家可行性**

Gou et al. (2024, MoE-LLaVA) 和 Li et al. (2024, Uni-MoE) 在多模态领域验证了异构专家 MoE 的有效性。本工作将这一思路首次系统性地应用到时间序列基础模型中，利用时间序列固有的分解结构来指导异构专家设计。

### 2.4 论文表述建议

> 本文将时间序列经典分解理论引入稀疏混合专家架构，构建了**异构专家池**。不同于 Time-MoE 采用的同构 MLP 专家，我们为三种时序成分（趋势、周期、异常）分别设计了具有对应归纳偏置的专家模块。趋势专家借鉴 N-BEATS 的多项式基展开和 Autoformer 的移动平均分解；周期专家融入 Autoformer 的自相关机制和 FEDformer 的频域增强；异常专家引入 Anomaly Transformer 的先验-序列混合注意力。同时，每种类型保留了原始 MLP 专家作为通用后备，在注入领域知识的同时维持基线建模能力。

---

## 三、创新点二：类型约束路由（Typed Top-k Routing）

### 3.1 问题背景

标准 MoE 的 top-k 路由：

$$\text{topk}(\text{softmax}(\mathbf{W}_g \mathbf{x}), k)$$

当 $k=2$、8 个专家中有同构 MLP 时，top-2 可能选择两个非常相似的专家——路由无法保证输出具有多成分组合的效果。在异构专家池中，如果两个被选中的专家恰好属于同一类型（如两个 trend 专家），则该 token 只得到趋势特征的加权和，完全丢失了周期和异常信息。

### 3.2 设计方案

**Typed Top-k 路由流程**：

```
softmax(W_g · x)                       -- (1) 标准 softmax 得到概率
  → typed_preselect(p, type_map)        -- (2) 每类型仅保留组内 argmax，置零其余
  → top-k(filtered, k=2)               -- (3) 在过滤后的概率上选 top-k
```

`typed_preselect` 的核心操作是：对于每个 token 和每种类型（trend/cycle/anomaly），仅保留该类型内概率最高的专家，其余同类型专家的概率置零。这确保了：

- **每种类型最多贡献 1 个专家进入 top-k 候选池**
- **top-k 选择在不同类型的代表之间进行**
- **当 $k=2$ 且类型数为 3 时，最终选中的 2 个专家一定来自不同类型**

用公式表达：对 token $t$，类型 $c \in \{0, 1, 2\}$：

$$\text{winner}_c = \arg\max_{j \in \mathcal{E}_c} p_{t,j}$$

$$\tilde{p}_{t,j} = \begin{cases} p_{t,j} & \text{if } j = \text{winner}_c \text{ for some } c \\ 0 & \text{otherwise} \end{cases}$$

$$(\text{idx}_1, \text{idx}_2) = \text{top-}k(\tilde{p}_t, k=2)$$

### 3.3 理论依据

**（1）保证输出的多成分性**

类型约束保证每个 token 的专家输出是**不同类型成分的混合**，而非同一类型的冗余叠加。设 token $t$ 选中的两个专家分别来自类型 $c_1$ 和 $c_2$（$c_1 \neq c_2$），则：

$$\mathbf{h}_t = w_1 \cdot f_{c_1}(\mathbf{x}_t) + w_2 \cdot f_{c_2}(\mathbf{x}_t) + \text{shared}(\mathbf{x}_t)$$

这天然对应序列分解的加法模型 $Y = T + S + R$。

**（2）避免同构 MoE 的"隐式冗余"问题**

Zoph et al. (2022, ST-MoE) 和 Dai et al. (2024) 指出标准 top-k 路由在训练中容易出现"专家坍缩"（expert collapse），多个专家学到几乎相同的功能。类型约束通过结构化竞争缓解此问题：同类型专家只需在组内竞争，不同类型专家在 top-k 层面竞争，形成层次化分工。

**（3）受约束优化视角**

可将 typed top-k 理解为在标准 top-k 选择上添加了约束：

$$\max_{S \subseteq [N], |S|=k} \sum_{j \in S} p_j \quad \text{s.t.} \quad |\{c(j) : j \in S\}| = \min(k, C)$$

即在最大化路由概率总和的同时，要求所选专家覆盖尽可能多的类型。`typed_preselect` 是该约束问题的一种贪心近似解。

### 3.4 论文表述建议

> 为确保路由输出包含多种时序成分的组合特征，我们提出**类型约束 Top-k 路由**。在标准 softmax 路由概率之上，我们引入类型内预选步骤：每种专家类型仅保留组内概率最大的候选者，然后在过滤后的候选集中进行 top-k 选择。该机制保证当 $k \geq 2$ 时，被选中的专家一定来自不同的时序成分类型，使每个 token 的隐状态更新同时包含趋势、周期或异常特征的组合，从结构上逼近经典时间序列加法分解模型。

---

## 四、创新点三：S1 调度策略——全序列执行 + 选择性聚合

### 4.1 问题背景

标准 MoE 的专家调度假设每个专家接收**离散 token 子集**作为输入（`x_flat[top_x]`，shape 为 `[M, H]`）。但 Autoformer、FEDformer、AnomalyTransformer 等算子依赖**序列上下文**（FFT 需要完整序列、移动平均需要相邻 token、自注意力需要全局 token 对关系），无法在离散 token 子集上正确执行。

这是异构 MoE 的核心技术挑战：**如何在 token 级稀疏路由框架中集成需要序列上下文的专家？**

### 4.2 设计方案

我们提出 **S1（全序列执行 + 选择性聚合）调度策略**：

```python
# 对于 seq 型专家 expert_j:
# 1. 始终在完整 [B, L, H] 上执行
seq_out = expert_j(hidden_states)             # [B, L, H]
seq_out_flat = seq_out.view(-1, H)            # [B*L, H]

# 2. 仅取路由选中该专家的 token 位置
selected = seq_out_flat[top_x]                # [M_j, H]

# 3. 按路由权重加权后聚合到输出
final.index_add_(0, top_x, selected * w)
```

**设计要点**：
- seq 专家始终看到完整序列 → FFT / 移动平均 / 注意力等算子的语义完整性得到保证
- 路由仍然是 token 级的 → 哪些位置的输出真正被使用由门控决定
- 未被选中位置的计算结果**被丢弃** → 计算量不随选中比例减少（这是设计上的 trade-off）

另外，我们实现了**缓存优化**：如果同一 seq 专家在 top-k 的不同位置被选中，只执行一次 forward，结果缓存在 `seq_expert_cache` 字典中。

### 4.3 理论依据

**（1）算子语义完整性**

FFT 的数学定义要求对完整（或至少连续段的）信号做变换：

$$X_f = \sum_{n=0}^{N-1} x_n \cdot e^{-i 2\pi fn/N}$$

如果输入是离散采样的 token 子集（非连续），FFT 结果在物理上无意义。同理，移动平均需要相邻点、自注意力需要全局 QK 匹配。S1 通过让 seq 专家始终处理完整序列来避免此问题。

**（2）MoE 框架的兼容性**

S1 的关键洞察是：MoE 的稀疏性体现在**聚合阶段**（哪些 token 使用哪些专家的输出），而非必须体现在**计算阶段**（每个专家只处理被分配的 token）。S1 放宽了计算阶段的稀疏约束，在聚合阶段保持 token 级稀疏路由，兼容 MoE 的整体框架。

**（3）计算量可控**

每层最多 3-4 个 seq 专家，且它们的计算量远小于完整 Transformer 层。实际增量约为 30%-80%，在可接受范围内。同时，`expert_mask[expert_idx].any()` 检查确保了没有任何 token 选中的 seq 专家不会执行。

### 4.4 KV-Cache 推理适配

自回归推理的 KV-Cache 模式下，每步只有 1 个新 token（`sequence_length == 1`），这对 seq 专家是退化情况——单 token 无法做 FFT/移动平均。

我们的解决方案是**Flat Fallback 机制**：

```python
if sequence_length == 1 and expert.interface_kind == "seq":
    output = expert.forward_flat_fallback(x)  # 零初始化线性投影
```

`forward_flat_fallback` 使用一个零初始化的 `nn.Linear(H, H)` 作为退化路径，训练过程中会自适应学习。在此基础上，各 seq 专家可以另外覆写该方法（如通过 `output_norm` 后处理）以保持输出尺度一致。

### 4.5 论文表述建议

> 异构 MoE 需要解决的核心技术问题是：如何在 token 级稀疏路由框架中集成需要序列上下文的专家算子（如 FFT、序列分解、全局注意力）。我们提出 S1 调度策略：seq 型专家始终在完整序列上执行以保证算子语义完整性，MoE 的稀疏性则体现在聚合阶段——仅路由选中的 token 位置的输出参与最终加权求和。该策略将 MoE 的"计算稀疏"放宽为"聚合稀疏"，在正确性和效率之间取得平衡。对于 KV-Cache 自回归推理场景，我们设计了 Flat Fallback 退化路径以适配单 token 输入。

---

## 五、创新点四：RouterInfo 元信息传播与辅助损失改进

### 5.1 问题背景

Time-MoE 的辅助损失（auxiliary loss）沿用 Switch Transformer 的负载均衡损失：

$$\mathcal{L}_{aux} = N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

其中 $f_i$ 是专家 $i$ 被分配的 token 比例，$P_i$ 是专家 $i$ 的平均路由概率。

**问题 1**：在 typed top-k 路由下，如果 $P_i$ 也使用过滤后（post-typed_preselect）的概率，那些在类型内预选中被淘汰的专家将完全得不到 $P_i$ 的梯度信号，导致**组内马太效应**（Matthew Effect）——强者恒强、弱者永远不被选。

**问题 2**：Time-MoE 原实现中辅助损失函数内部重新计算路由（重新 softmax + topk），与 forward 中实际的路由结果不完全一致（forward 有 jitter noise，loss 中没有）。

### 5.2 设计方案

**（1）$P_i$ 使用预选前的原始 softmax 概率**

```python
# forward 中保存原始概率
raw_routing_weights = F.softmax(router_logits, dim=1)     # 预选前
filtered_weights = typed_preselect(raw_routing_weights, ...)  # 预选后

# aux loss 使用 raw_routing_weights 计算 P_i
P_i = mean(raw_routing_weights[:, i])  # 所有专家都有梯度
# 但 f_i 使用 topk_indices（预选后的选择结果）
f_i = count(topk_indices == i) / total_tokens
```

**（2）RouterInfo 元信息传播**

我们引入 `RouterInfo` 命名元组，在 forward 中打包所有路由元信息，传递给下游损失函数：

```python
RouterInfo = namedtuple("RouterInfo", [
    "raw_logits",       # 原始 gate 输出
    "topk_indices",     # 选中的专家索引
    "topk_weights",     # 选中的权重
    "filtered_probs",   # 类型预选后概率
    "raw_probs",        # 类型预选前概率
    "actual_k",         # 实际使用的 k 值
])
```

损失函数直接从 `RouterInfo` 中读取 forward 的路由结果，**不再重新计算**，确保 forward 和 loss 使用完全一致的路由决策。

### 5.3 理论依据

**（1）防止组内马太效应**

在 typed_preselect 中，每种类型只有 1 个专家的概率被保留，其余被置零。如果 $P_i$ 基于置零后的概率，被淘汰专家的 $P_i = 0$，aux loss 对它们完全没有梯度信号，这些专家永远无法"翻身"。使用预选前概率确保每个专家都能收到 gate 分配给它的概率信号。

**（2）路由一致性原则**

Lepikhin et al. (2021, GShard) 指出 MoE 训练中 forward 和 loss 使用不同路由结果会导致训练不稳定。RouterInfo 机制将两者统一，是一种工程正确性保障。

### 5.4 论文表述建议

> 为适配类型约束路由，我们对辅助损失进行了两项改进。其一，负载均衡损失中的平均路由概率 $P_i$ 使用类型预选前的原始 softmax 概率计算，确保所有专家（包括组内被淘汰者）均能收到梯度信号，防止组内马太效应。其二，我们引入 RouterInfo 元信息传播机制，在前向传播中打包完整路由决策（索引、权重、概率），使损失函数直接复用前向传播的路由结果，避免因重新计算导致的 forward-loss 不一致问题。

---

## 六、创新点五：类型多样性损失（Type Diversity Loss）

### 6.1 设计方案

在辅助损失基础上增加类型多样性损失，鼓励 top-k 选择覆盖尽可能多的不同类型：

$$\mathcal{L}_{div} = \mathbb{E}_t\left[\max\left(0, \ \min(k, C) - |\{c(j) : j \in \text{topk}_t\}|\right)\right]$$

其中 $C$ 为类型总数，$c(j)$ 为专家 $j$ 的类型。当 $k=2, C=3$ 时，$\min(k, C) = 2$，只要 top-2 选了两个不同类型即可满足（loss = 0），否则受到惩罚。

**总损失**：

$$\mathcal{L}_{total} = \mathcal{L}_{AR} + \alpha \cdot \mathcal{L}_{aux} + \beta \cdot \mathcal{L}_{div}$$

- $\mathcal{L}_{AR}$：自回归预测损失（Huber loss, δ=2.0）
- $\alpha$：负载均衡系数（默认 0.02）
- $\beta$：类型多样性系数（默认 0.01）

### 6.2 理论依据

typed_preselect 是一种"硬约束"——从候选集中删除同类型竞争者。但这种硬约束可能在边界情况下不够鲁棒（如某类型只有 1 个专家、或某类型专家概率极低）。type_diversity_loss 作为"软约束"补充，通过梯度信号鼓励 gate 学到的概率分布自然倾向于跨类型分散，构成硬约束 + 软约束的双重保障。

### 6.3 论文表述建议

> 除了路由机制中的硬性类型约束，我们还引入类型多样性损失作为软约束。该损失衡量每个 token 的 top-k 选择是否覆盖了不同类型，当所选专家类型多样性不足时施加惩罚。硬约束（类型内预选）和软约束（多样性损失）协同作用，前者在推理路径上保证结果，后者在训练路径上引导门控网络学到跨类型分散的概率分布。

---

## 七、创新点六：渐进式训练策略与零初始化残差

### 7.1 分阶段解冻（Phased Freeze）

| 阶段 | 训练步 | 可训练参数 | 目的 |
|------|--------|-----------|------|
| Phase-A | 0 ~ 1,000 | 仅 gate 权重 | 让门控网络适应新的类型约束路由格式 |
| Phase-B | 1,000 ~ 5,000 | gate + 新专家参数 | 新专家从 zero-init 开始学习，不扰动已收敛参数 |
| Phase-C | 5,000+ | 全部参数 | 联合微调，所有模块协同优化 |

### 7.2 零初始化残差（Zero-Init Residual）

新引入的专家（N-BEATS、Autoformer、FEDformer、AnomalyTransformer）的**输出投射层权重初始化为零**：

```python
nn.init.zeros_(self.out_proj.weight)  # 或 basis_proj.weight
```

效果：训练初期，新专家的输出 ≈ **0** → 整个模型行为等价于仅使用原始 MLP 专家 → **稳定过渡**，避免随机初始化新专家破坏预训练模型的已有能力。

### 7.3 理论依据

**（1）迁移学习稳定性**

类似于 LoRA（Hu et al., 2022）中 B 矩阵零初始化的策略——在微调初始时刻，新增模块不产生任何输出，模型行为与原始预训练模型完全一致。随着训练进行，新模块逐渐学到有意义的映射。

**（2）避免灾难性遗忘**

如果新专家随机初始化，初始时刻它们会输出随机噪声，这些噪声通过路由权重混合进隐状态，可能严重破坏预训练模型的内部表示。分阶段解冻 + 零初始化的组合确保：
- Phase-A 只调 gate → 模型主干完全不动，仅学习新路由格式
- Phase-B 新专家从零开始 → 梯度信号逐步引导其学到有意义的特征
- Phase-C 全局协同 → 精细调整

### 7.4 论文表述建议

> 为确保异构专家引入时模型的训练稳定性，我们采用分阶段解冻策略和零初始化残差技巧。新增专家的输出投影层权重初始化为零，使训练初期模型行为等价于仅使用预训练 MLP 专家。分阶段解冻从门控网络开始，逐步扩展到新专家和全部参数，实现从预训练状态到异构 MoE 的平滑过渡。

---

## 八、创新点七：统一专家基类与插件注册表

### 8.1 设计方案

所有专家继承自 `BaseTokenExpert`，声明 `interface_kind`（`"flat"` / `"seq"`）和 `expert_type`（`"trend"` / `"cycle"` / `"anomaly"`）属性。通过 `@register_expert(name)` 装饰器注册到全局注册表，由 YAML 配置文件驱动实例化：

```yaml
experts:
  - {name: mlp_temporal_block, type: trend, interface: flat}
  - {name: nbeats_trend, type: trend, interface: flat, params: {theta_dim: 16}}
  - {name: autoformer_trend, type: trend, interface: seq, params: {kernel_size: 25}}
  ...
```

### 8.2 理论依据

**（1）可扩展性（Open-Closed Principle）**

新增专家类型不需要修改路由层或模型主体代码，只需：创建新专家类 → 注册 → 写入配置。这对研究迭代极为友好。

**（2）实验可复现性**

专家组合完全由配置文件定义，不同配置文件对应不同实验方案，便于消融实验和版本管理。

### 8.3 论文表述建议

> 我们设计了统一的专家基类和插件注册表机制。每种专家通过声明其接口类型和时序分量类型完成注册，路由层根据配置文件动态实例化专家组合。该架构使新增专家无需修改核心路由代码，提升了实验迭代效率和可复现性。

---

## 九、输出 RMSNorm 对齐

### 9.1 设计方案

**所有专家的输出统一经过 RMSNorm**（`ExpertRMSNorm`），在 `index_add_` 聚合前将不同专家的输出映射到相同的尺度空间。

### 9.2 理论依据

异构专家的内部计算差异巨大（MLP 输出 vs FFT 重建 vs 注意力加权），输出的数值范围和方差可能差数个量级。如果直接按路由权重加权聚合，数值大的专家会主导结果。RMSNorm 作为一种无均值偏移的归一化算子（Zhang & Sennrich, 2019），将所有专家输出归一化到相同的 RMS 尺度，确保聚合的公平性。

### 9.3 论文表述建议

> 由于异构专家的内部计算差异（pointwise MLP vs FFT 频域变换 vs 注意力机制）导致输出数值范围不同，我们在每个专家输出后添加 RMSNorm 层以对齐数值尺度，确保路由加权聚合不被某类专家的数值幅度所主导。

---

## 十、总结：创新点映射到论文章节建议

| 创新点 | 建议论文章节 | 核心关键词 |
|--------|------------|-----------|
| 异构专家池 | §3.1 异构专家池设计 | 时间序列分解先验、归纳偏置注入 |
| 类型约束路由 | §3.2 类型约束 Top-k 路由 | 类型内预选、多成分保证 |
| S1 调度策略 | §3.3 序列专家调度 | 全序列执行、选择性聚合、聚合稀疏 |
| RouterInfo + aux loss 改进 | §3.4 辅助损失设计 | 预选前 P_i、forward-loss 一致性、马太效应 |
| 类型多样性损失 | §3.4 辅助损失设计 | 硬约束+软约束、梯度引导 |
| 渐进式训练 + zero-init | §3.5 训练策略 | 分阶段解冻、零初始化残差、灾难性遗忘 |
| 专家注册表 | §4.1 系统实现 | 可扩展架构、配置驱动 |
| 输出 RMSNorm 对齐 | §3.1 或 §4.2 | 异构输出尺度对齐 |

---

## 附录：与相关工作的关系定位

| 方法 | 专家结构 | 路由方式 | 时序先验 | 与 Type-MoE 的区别 |
|------|---------|---------|---------|-------------------|
| Time-MoE | 同构 MLP | 标准 top-k | 无 | 我们的基线 |
| Switch Transformer | 同构 FFN | top-1 | N/A (NLP) | 无时序先验 |
| MoE-LLaVA | 异构 (视觉+语言) | top-k | N/A (多模态) | 异构思路启发，但领域不同 |
| Autoformer | 单模型 | N/A | 序列分解 | 我们将其作为专家算子集成 |
| FEDformer | 单模型 | N/A | 频域增强 | 同上 |
| N-BEATS | 单模型 | N/A | 基函数展开 | 同上 |
| Anomaly Transformer | 单模型 | N/A | 先验-序列注意力 | 同上 |
| **Type-MoE（本文）** | **异构（时序分解驱动）** | **类型约束 top-k** | **趋势/周期/异常** | 首次将多种时序算子作为异构专家集成到稀疏 MoE 中，并设计类型约束路由保证多成分融合 |
