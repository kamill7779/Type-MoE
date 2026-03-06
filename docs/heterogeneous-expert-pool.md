# 专家结构改进：从同构 MLP 到异构专家池

> 更新时间：2026-03-06  
> 适用代码仓库：`C:\Users\kamil.liu\source\Type-MoE`  
> 对应实现文件：`time_moe/models/experts/`、`time_moe/models/modeling_time_moe.py`

---

## 4.1 原始专家结构（Homogeneous MLP Experts）

Time-MoE 的所有专家均为完全相同结构的 `TimeMoeTemporalBlock`，其定义如下：

```python
class TimeMoeTemporalBlock(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act):
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn    = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

这是标准的**门控 MLP（Gated MLP / SwiGLU 变体）**，其本质是对每个 token 的隐状态向量做 pointwise 非线性变换：

$$f(\mathbf{x}) = \mathbf{W}_{\text{down}} \cdot \left(\sigma(\mathbf{W}_{\text{gate}} \mathbf{x}) \odot \mathbf{W}_{\text{up}} \mathbf{x}\right)$$

Time-MoE 将 $N=8$ 个完全相同结构（仅初始化不同）的此类模块并列组成专家池，配合 Top-k 路由进行稀疏分发。

---

## 4.2 同构专家的根本局限

### 问题一：无法主动建模时序结构（缺乏归纳偏置）

`TimeMoeTemporalBlock` 是纯 pointwise 变换——**它独立处理每个 token 的隐向量，不在时间维度上建立任何显式依赖**。它唯一能利用的时序信息，是由上游 Self-Attention 编码后通过残差传递下来的隐式表示。

时间序列天然具备三种可分解的结构性成分：

$$Y_t = T_t + S_t + R_t$$

其中 $T_t$ 为低频趋势，$S_t$ 为周期/季节分量，$R_t$ 为高频残差（含异常）。同构 MLP 对这三种成分的归纳偏置完全为零——它没有移动平均算子（无法提取趋势），没有频域变换（无法提取周期），没有异常距离先验（无法区分正常与异常关联）。所有的"分工"只能完全依赖数据驱动的隐式学习。

### 问题二：同质化风险（Expert Collapse）

实际训练中，8 个专家都是同一结构、同一训练目标，仅依靠随机初始化的细微差异来驱动专业化分工。这导致：

- **强者恒强（马太效应）**：初始阶段，某几个专家因随机初始化偶然得到更高的 gate 分数，被持续选中、持续更新；其他专家长期被忽视；
- **冗余收敛**：最终多个专家学到几乎相同的特征映射，参数利用率低，MoE 的容量优势大量浪费；
- **Zoph et al.（2022, ST-MoE）和 Dai et al.（2024）** 均在实验中观察到了严重的专家同质化现象，并指出这是同构 MoE 在多样化任务上效率低下的主要原因之一。

### 问题三：构建异构路由后专家结构错配

若在不修改专家结构的前提下直接引入类型约束路由（typed top-k），将趋势类、周期类、异常类 token 分别导向不同专家，但这些专家仍是相同的 MLP block，则：

- 类型标签形同虚设，专家无法发挥对应类型的建模优势；
- 路由施加了结构约束，专家却无结构响应，形成**路由-专家的语义断层**。

---

## 4.3 改进方案：异构专家池（Heterogeneous Expert Pool）

### 4.3.1 设计思想

本文将8个专家槽按时序分解理论划分为**三种类型**，每种类型内部同时保留原始 MLP 专家（维持基线兼容性）和引入具有领域归纳偏置的新型专家（注入专业化能力）。

完整专家池配置如下：

| 编号 | 专家名 | 类型 | 接口 | 核心算子 | 零初始化 | 归纳偏置 |
|------|--------|------|------|---------|---------|---------|
| E0 | `mlp_temporal_block` | trend | flat | Gated MLP | ✗（复用预训练权重） | 通用非线性 |
| E1 | `nbeats_trend` | trend | flat | FC Stack → θ → 基函数投影 | ✓ | 多项式展开，低频趋势 |
| E2 | `autoformer_trend` | trend | seq | 移动平均序列分解 | ✓ | 趋势平滑提取 |
| E3 | `mlp_temporal_block` | cycle | flat | Gated MLP | ✗（复用预训练权重） | 通用非线性 |
| E4 | `autoformer_cycle` | cycle | seq | 序列分解 + FFT 自相关 | ✓ | 周期对齐模式捕获 |
| E5 | `fedformer_cycle` | cycle | seq | FFT → 可学习频域权重 → IFFT | ✓ | 低频频域周期建模 |
| E6 | `mlp_temporal_block` | anomaly | flat | Gated MLP | ✗（复用预训练权重） | 通用非线性 |
| E7 | `anomaly_attn` | anomaly | seq | 高斯先验 + 序列自注意力混合 | ✓ | 正常关联 vs 异常偏离 |

### 4.3.2 各新型专家的算子设计

#### N-BEATS 趋势专家（E1，flat 接口）

灵感来自 Oreshkin et al.（2020, N-BEATS），将隐状态经过多层全连接堆叠压缩为低维系数 $\boldsymbol{\theta}$，再投影回隐空间。其关键在于通过**瓶颈降维**（$H \to \theta_{\text{dim}} \to H$，$\theta_{\text{dim}} \ll H$）迫使网络提取低频平滑特征：

$$\mathbf{h} = \text{FC}_{\text{stack}}(\mathbf{x}) \in \mathbb{R}^d,\quad \boldsymbol{\theta} = \mathbf{W}_\theta \mathbf{h} \in \mathbb{R}^{\theta_{\text{dim}}},\quad \mathbf{out} = \mathbf{W}_{\text{basis}} \boldsymbol{\theta}$$

零初始化 $\mathbf{W}_{\text{basis}}$，确保训练初期输出为零，不干扰预训练表示：

```python
def zero_init_output(self):
    nn.init.zeros_(self.basis_proj.weight)
```

#### Autoformer 趋势专家（E2，seq 接口）

直接移植 Wu et al.（2021, Autoformer）的序列分解层。以移动平均核 $\text{AvgPool1d}(k=25)$ 从完整序列中提取趋势分量，残差为季节分量：

$$T_t = \text{MovingAvg}(\mathbf{X}),\quad S_t = \mathbf{X} - T_t$$

E2 取趋势分量 $T_t$ 经线性投影后输出：

$$\mathbf{out} = \mathbf{W}_{\text{proj}} \cdot T_t$$

该算子需要完整的序列上下文（边界填充依赖序列首尾值），因此接口类型为 `seq`。

#### Autoformer 周期专家（E4，seq 接口）

取序列分解后的季节分量 $S_t$，再经过**简化自相关（Simplified AutoCorrelation）**，在频域中保留能量最强的 Top-$k$ 频率分量：

$$X_f = \text{FFT}(S_t),\quad \tilde{X}_f = X_f \cdot \mathbb{1}[\text{freq} \in \text{Top-}k_{\text{freq}}],\quad \mathbf{out} = \mathbf{W}_\text{proj} \cdot \text{IFFT}(\tilde{X}_f)$$

通过谱滤波保留主要的周期振型，抑制高频噪声。

#### FEDformer 周期专家（E5，seq 接口）

灵感来自 Zhou et al.（2022, FEDformer）。不同于 E4 的硬性频率筛选，E5 在频域上施加**可学习的逐频率权重**：

$$\mathbf{X}_f = \text{FFT}(\mathbf{X}),\quad \tilde{\mathbf{X}}_f[:m, :] = \mathbf{X}_f[:m, :] \odot \mathbf{W}_{\text{freq}},\quad \mathbf{out} = \mathbf{W}_\text{proj} \cdot \text{IFFT}(\tilde{\mathbf{X}}_f)$$

其中 $\mathbf{W}_{\text{freq}} \in \mathbb{R}^{m \times H}$ 是可训练参数（$m$ 为截断频率数），使模型能够自适应学习每个频率分量的重要性，而非依赖固定阈值。

#### AnomalyTransformer 异常专家（E7，seq 接口）

灵感来自 Xu et al.（2022, Anomaly Transformer），将经典自注意力与高斯距离先验混合，建模"正常关联模式"：

$$\text{Prior}_{ij} = \mathcal{N}(|i-j|;\ 0,\ \sigma_i^2),\quad \text{Series}_{ij} = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)_{ij}$$

$$\text{Attn} = 0.5 \cdot \text{Series} + 0.5 \cdot \text{Prior},\quad \mathbf{out} = \mathbf{W}_\text{proj} \cdot (\text{Attn} \cdot V)$$

其中 $\sigma_i$ 由网络从输入学习得到。**正常时间点受高斯先验约束，只与邻近位置强相关；异常时间点则因跳离先验而在混合注意力分布中产生偏差。** 这一先验天然编码了"正常时序连续平滑"的假设。

---

## 4.4 接口统一与插件注册机制

### 4.4.1 统一基类 `BaseTokenExpert`

所有专家继承自 `BaseTokenExpert`，通过两个类属性声明自身特性：

```python
class BaseTokenExpert(nn.Module):
    expert_type: str = "generic"    # trend / cycle / anomaly
    interface_kind: str = "flat"    # flat | seq
```

- `flat` 专家接受 `[M, H]` 形状（M 为被路由到该专家的 token 数），支持 token 级稀疏执行；
- `seq` 专家接受 `[B, L, H]` 完整序列，需要全局上下文（FFT、移动平均、全局注意力均依赖此）。

### 4.4.2 插件注册表（Expert Registry）

通过 `@register_expert("name")` 装饰器将专家类注册到全局字典，由 `build_expert(spec, ...)` 函数按 YAML 配置动态实例化：

```python
# 注册示例
@register_expert("nbeats_trend")
class NBeatsTokenExpert(BaseTokenExpert): ...

# 构建示例（由 YAML spec 驱动）
expert = build_expert(
    spec={"name": "nbeats_trend", "type": "trend", "zero_init_output": True,
          "params": {"num_layers": 4, "theta_dim": 16}},
    hidden_size=384, intermediate_size=192, hidden_act="silu"
)
```

这使得专家组合完全由 `configs/typed_experts/minimal_test.yaml` 配置文件定义，新增专家无需修改任何路由或模型主体代码（**Open-Closed 原则**）。

### 4.4.3 `TimeMoeSparseExpertsLayer` 中的切换逻辑

模型代码中，当 `custom_expert_specs` 非空时，专家池由注册机制构建；否则回退到原始同构 MLP：

```python
if isinstance(custom_expert_specs, list) and len(custom_expert_specs) > 0:
    self.experts = nn.ModuleList([
        build_expert(spec, hidden_size=..., ...) for spec in custom_expert_specs
    ])
else:
    # 原始同构 MLP 专家（Time-MoE 基线）
    self.experts = nn.ModuleList([
        TimeMoeTemporalBlock(hidden_size=..., ...) for _ in range(self.num_experts)
    ])
```

此设计保证了对原始 Time-MoE 流程的**完全向后兼容**。

---

## 4.5 输出尺度对齐：ExpertRMSNorm

由于不同专家的内部计算差异极大（Gated MLP 输出 vs FFT 重建信号 vs 自注意力加权和），各专家输出的数值范围和方差差距可能达数个量级。若直接按路由权重加权聚合，数值幅度大的专家将主导结果，抹去其他专家的贡献。

为此，所有专家输出在回填至全局隐状态前，统一经过 `ExpertRMSNorm`：

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \cdot \boldsymbol{\gamma}, \quad \text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{H}\sum_{i=1}^H x_i^2}$$

这是 Zhang & Sennrich（2019）提出的无均值偏移归一化算子，将不同专家输出归一化到相同的 RMS 尺度，确保路由加权聚合的公平性。

```python
class ExpertRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        return self.weight * x * torch.rsqrt(var + self.eps)
```

---

## 4.6 零初始化残差（Zero-Init Residual）

5 个新引入的异构专家（E1/E2/E4/E5/E7）的**输出投影层权重均初始化为零**。以 E2 为例：

```python
def zero_init_output(self):
    nn.init.zeros_(self.out_proj.weight)   # AutoformerTrendExpert
```

**效果**：训练初期，新专家的输出恒等于 $\mathbf{0}$，整个模型行为完全等价于仅使用预训练 MLP 专家（E0/E3/E6）。随着训练推进，梯度逐步流入这些零权重层，新专家的输出从零开始增长，实现对预训练模型表示的**无破坏渐进融合**。

这与 Hu et al.（2022, LoRA）中 B 矩阵零初始化的策略在原理上完全一致——在微调初始时刻，新增模块不产生任何干扰，模型行为与原始预训练状态严格等同。

---

## 4.7 S1 调度策略：seq 专家的稀疏路由适配

标准 MoE 假设专家接受**离散的 token 子集**。但 FFT、移动平均、全局注意力等算子要求**完整序列上下文**，二者天然冲突。

本文提出 **S1（全序列执行 + 选择性聚合）调度策略**，解决对 `seq` 型专家的稀疏路由适配问题：

```
seq 专家 forward_seq([B, L, H])
   ├─ 输出 full_out: [B, L, H]
   ├─ 展平: full_out_flat = full_out.view(B*L, H)
   ├─ 取路由选中的 token 位置: selected = full_out_flat[top_x]
   └─ 加权回填: final.index_add_(0, top_x, selected * routing_weight)
```

**关键权衡**：
- seq 专家始终做完整序列计算，算子语义完整性得到保证；
- 最终只有被路由选中的 token 位置的输出参与聚合，路由的稀疏性体现在聚合阶段而非计算阶段；
- 未被选中的计算结果被丢弃，引入一定额外计算开销（这是已知 trade-off）。

同一专家若在 Top-k 的多个位置均被选中，仅执行一次 forward 并通过 `seq_expert_cache` 缓存结果，避免重复计算。

---

## 4.8 新旧专家体系完整对比

| 维度 | 原始 Time-MoE | Type-MoE（本文） |
|------|--------------|----------------|
| **专家结构** | 8 个同构 `TimeMoeTemporalBlock`（Gated MLP） | 3 种 MLP（复用预训练）+ 5 种异构新专家 |
| **专家接口** | 仅 flat（token 级 `[M, H]`） | flat + seq（完整序列 `[B, L, H]`） |
| **归纳偏置** | 无（纯 pointwise 非线性） | 趋势（MA/瓶颈降维）/ 周期（FFT/自相关）/ 异常（高斯先验注意力） |
| **初始化** | 随机初始化 | 新专家零初始化输出投影（zero-init residual） |
| **输出对齐** | 无（直接聚合） | ExpertRMSNorm 对齐尺度 |
| **专家扩展性** | 写死在 `ModuleList`，需改模型代码 | 插件注册表 + YAML 配置驱动，Open-Closed |
| **与路由的耦合** | 路由同构，专家无类型语义 | 路由类型与专家归纳偏置严格对应 |
| **向后兼容** | — | `custom_expert_specs` 为空时自动回退到原始 MLP 专家 |

---

## 4.9 理论依据总结

**（1）时间序列分解先验（Decomposition Theory）**

经典信号分析理论（STL, Cleveland et al., 1990; MSTL, Bandara et al., 2021）证明将时间序列建模为 $Y_t = T_t + S_t + R_t$ 在实践中是有效的先验假设。本文将该先验直接注入专家结构设计，使 MoE 的"软分工"转变为带有领域约束的"硬分工"，从根本上提升了专业化的可靠性和效率。

**（2）异构专家可行性（Heterogeneous MoE）**

Gou et al.（2024, MoE-LLaVA）和 Li et al.（2024, Uni-MoE）在视觉-语言多模态领域验证了异构专家（不同专家处理不同模态）可显著提升 MoE 的能力边界。本文将该思路首次系统性地迁移到时间序列基础模型中，利用时序固有的分解结构来指导专家异构化设计，逻辑闭环更为紧密。

**（3）渐进融合稳定性（Progressive Integration）**

零初始化残差消除了新专家初始阶段对预训练模型的干扰，是 LoRA（Hu et al., 2022）、ReLoRA（Lialin et al., 2023）等主流参数高效微调方法已验证的稳健机制。分阶段解冻（Phased Freeze）进一步在优化阶层面强化了渐进融合的稳定性。

**（4）输出对齐必要性**

RMSNorm 对跨专家输出尺度的对齐，在 MoA（Mixture of Adapters，Zhang et al., 2024）和 MoH（Mixture of Heads）等近期工作中已被证明是异构聚合场景下的关键稳定手段，否则数值幅度优势的专家将形成新的"马太垄断"。
