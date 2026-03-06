对 `Time-MoE` 进行如下改造，改造版本简称 `Type-MoE`：

### 1. 引入异构专家：

#### 1.1 原始做法

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

MLP模块，其本质是对每个 token 的隐状态向量做 pointwise 非线性变换：

$$f(\mathbf{x}) = \mathbf{W}_{\text{down}} \cdot \left(\sigma(\mathbf{W}_{\text{gate}} \mathbf{x}) \odot \mathbf{W}_{\text{up}} \mathbf{x}\right)$$

其中 $\mathbf{x} \in \mathbb{R}^H$ 为 token 隐状态向量；$\mathbf{W}_{\text{gate}}, \mathbf{W}_{\text{up}} \in \mathbb{R}^{d_i \times H}$ 与 $\mathbf{W}_{\text{down}} \in \mathbb{R}^{H \times d_i}$ 为可训练权重矩阵；$\sigma$ 为激活函数（SiLU），$\odot$ 表示逐元素乘法，$d_i$ 为中间层维度（Time-MoE 默认 $d_i = 4H = 1536$）。

Shazeer, N. (2020). *GLU variants improve transformer* [Preprint]. arXiv. [https://arxiv.org/abs/2002.05202](vscode-file://vscode-app/c:/Users/kamil.liu/AppData/Local/Programs/Microsoft VS Code/072586267e/resources/app/out/vs/code/electron-browser/workbench/workbench.html)

Time-MoE 将 $N=8$ 个完全相同结构（仅初始化不同）的此类模块并列组成专家池，配合 Top-k 路由进行稀疏分发。

Shi, X., Wang, S., Nie, Y., Li, D., Ye, Z., Wen, Q., & Jin, M. (2024). *Time-MoE: Billion-scale time series foundation models with mixture of experts*. arXiv. https://arxiv.org/abs/2409.16040

#### 1.2 可能存在的问题

##### 1.2.1 专家同质化

实际训练中，8 个专家都是同一结构、同一训练目标，仅依靠随机初始化的细微差异来驱动专业化分工，这会导致：

- 初始阶段，某几个专家因随机初始化偶然得到更高的 gate 分数，被持续选中、持续更新；其他专家长期被忽视；
- 最终多个专家学到几乎相同的特征映射，参数利用率低；

Zoph, B., Bello, I., Kumar, S., Du, N., Huang, Y., Dean, J., Shazeer, N., & Fedus, W. (2022). *ST-MoE: Designing stable and transferable sparse expert models*. arXiv. https://arxiv.org/abs/2202.08906

Dai, Z., et al. (2024). *On the efficiency of mixture-of-experts models*. arXiv.

##### 1.2.2 缺乏归纳偏置

时间序列天然具备三种可分解的结构性成分：

$$Y_t = T_t + S_t + R_t$$

其中 $T_t$ 为低频趋势，$S_t$ 为周期/季节分量，$R_t$ 为高频残差（含异常）。

Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). STL: A seasonal-trend decomposition procedure based on loess. *Journal of Official Statistics, 6*(1), 3–73.

同构 MLP 对这三种成分的归纳偏置完全为零——它没有移动平均算子（无法提取趋势），没有频域变换（无法提取周期），没有异常距离先验（无法区分正常与异常关联）。所有的分工只能完全依赖数据驱动的隐式学习。

#### 1.3 改进

保留原本的 `MLP` 模块，增加一些异构专家模块：

完整专家池配置如下：

| 编号 | 专家名               | 类型    | 接口 | 核心算子                    | 零初始化            |
| ---- | -------------------- | ------- | ---- | --------------------------- | ------------------- |
| E0   | `mlp_temporal_block` | trend   | flat | Gated MLP                   | ✗（复用预训练权重） |
| E1   | `nbeats_trend`       | trend   | flat | FC Stack → θ → 基函数投影   | ✓                   |
| E2   | `autoformer_trend`   | trend   | seq  | 移动平均序列分解            | ✓                   |
| E3   | `mlp_temporal_block` | cycle   | flat | Gated MLP                   | ✗（复用预训练权重） |
| E4   | `autoformer_cycle`   | cycle   | seq  | 序列分解 + FFT 自相关       | ✓                   |
| E5   | `fedformer_cycle`    | cycle   | seq  | FFT → 可学习频域权重 → IFFT | ✓                   |
| E6   | `mlp_temporal_block` | anomaly | flat | Gated MLP                   | ✗（复用预训练权重） |
| E7   | `anomaly_attn`       | anomaly | seq  | 高斯先验 + 序列自注意力混合 | ✓                   |

##### E1 趋势专家：N-BEATS

将隐状态经过多层全连接堆叠压缩为低维系数 $\boldsymbol{\theta}$，再投影回隐空间。其关键在于通过**瓶颈降维**（$H \to \theta_{\text{dim}} \to H$，$\theta_{\text{dim}} \ll H$）迫使网络提取低频平滑特征：

$$\mathbf{h} = \text{FC}_{\text{stack}}(\mathbf{x}) \in \mathbb{R}^d,\quad \boldsymbol{\theta} = \mathbf{W}_\theta \mathbf{h} \in \mathbb{R}^{\theta_{\text{dim}}},\quad \mathbf{out} = \mathbf{W}_{\text{basis}} \boldsymbol{\theta}$$

其中 $\mathbf{x} \in \mathbb{R}^H$ 为输入 token 隐状态；$\text{FC}_{\text{stack}}$ 为多层全连接堆叠（$H \to H \to \cdots$）；$\mathbf{W}_\theta \in \mathbb{R}^{\theta_{\text{dim}} \times H}$ 为瓶颈压缩矩阵，$\theta_{\text{dim}}=16 \ll H$ 迫使网络提取低频平滑特征；$\mathbf{W}_{\text{basis}} \in \mathbb{R}^{H \times \theta_{\text{dim}}}$ 为基函数投影矩阵，零初始化与预训练特征平滑叠加。

```python
# time_moe/models/experts/nbeats_token_expert.py
class NBeatsTokenExpert(BaseTokenExpert):
    def __init__(self, hidden_size, theta_dim=16, num_layers=4, ...):
        self.stack      = nn.Sequential(nn.Linear(H, H), nn.ReLU(), ...)  # FC Stack
        self.theta_proj = nn.Linear(hidden_size, theta_dim, bias=False)   # 投影到 θ 空间
        self.basis_proj = nn.Linear(theta_dim, hidden_size, bias=False)   # 基函数投影回隐空间

    def forward_flat(self, x):           # x: [M, H]
        h     = self.stack(x)            # [M, H]
        theta = self.theta_proj(h)       # [M, θ_dim]  ← 瓶颈压缩
        return self.norm(self.basis_proj(theta))  # [M, H]

    def zero_init_output(self):
        nn.init.zeros_(self.basis_proj.weight)    # 零初始化: 训练起点贡献为 0
```

Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2019). N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. In *Proceedings of the 8th International Conference on Learning Representations (ICLR 2020)*.

##### E4 周期/E2 趋势专家：Autoformer

直接移植 Autoformer 的序列分解层。以移动平均核 $\text{AvgPool1d}(k=25)$ 从完整序列中提取趋势分量，残差为季节分量：

$$T_t = \text{MovingAvg}(\mathbf{X}),\quad S_t = \mathbf{X} - T_t$$

其中 $\mathbf{X} \in \mathbb{R}^{L \times H}$ 为输入序列隐状态，$L$ 为序列长度，内核长度 $k=25$；$T_t$ 为低频趋势分量，$S_t = \mathbf{X} - T_t$ 为季节/残差分量。

E2 取趋势分量 $T_t$ 经线性投影后输出：

$$\mathbf{out} = \mathbf{W}_{\text{proj}} \cdot T_t$$

其中 $\mathbf{W}_{\text{proj}} \in \mathbb{R}^{H \times H}$ 为输出投影矩阵，零初始化保证训练起点对预训练特征不产生干扰。

该算子需要完整的序列上下文

取序列分解后的季节分量 $S_t$，再经过 Simplified AutoCorrelation，在频域中保留能量最强的 Top-$k$ 频率分量：

$$X_f = \text{FFT}(S_t),\quad \tilde{X}_f = X_f \cdot \mathbb{1}[\text{freq} \in \text{Top-}k_{\text{freq}}],\quad \mathbf{out} = \mathbf{W}_\text{proj} \cdot \text{IFFT}(\tilde{X}_f)$$

其中 $X_f$ 为季节分量的频域表示，$k_{\text{freq}}=3$ 为保留的主要频率分量数，$\mathbb{1}[\cdot]$ 为频率遮罩算子，仅保留能量最强的 Top-$k$ 频率。

```python
# time_moe/models/experts/autoformer_token_expert.py
class SeriesDecomposition(nn.Module):
    def forward(self, x):      # x: [B, L, H]
        trend    = self.moving_avg(x)    # 移动平均提取趋势 T_t
        seasonal = x - trend             # 残差即季节分量 S_t
        return trend, seasonal

class SimplifiedAutoCorrelation(nn.Module):
    def forward(self, x):      # x: [B, L, H]
        x_fft   = torch.fft.rfft(x, dim=1)          # 频域表示
        power   = x_fft.abs().mean(dim=(0, 2))       # 计算各频率能量
        top_idx = torch.topk(power, k=self.top_k_freq).indices  # Top-k 频率
        out_fft = torch.zeros_like(x_fft)
        out_fft[:, top_idx, :] = x_fft[:, top_idx, :]           # 戳除其他频率
        return torch.fft.irfft(out_fft, n=x.shape[1], dim=1)    # IFFT 还原
```

Wu, H., Xu, J., Wang, J., & Long, M. (2021). Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. Advances in Neural Information Processing Systems, 34, 13489–13500.

##### E5 周期专家：FEDformer

在频域上施加**可学习的逐频率权重**：

$$\mathbf{X}_f = \text{FFT}(\mathbf{X}),\quad \tilde{\mathbf{X}}_f[:m, :] = \mathbf{X}_f[:m, :] \odot \mathbf{W}_{\text{freq}},\quad \mathbf{out} = \mathbf{W}_\text{proj} \cdot \text{IFFT}(\tilde{\mathbf{X}}_f)$$

其中 $\mathbf{W}_{\text{freq}} \in \mathbb{R}^{m \times H}$ 是可训练参数（$m$ 为截断频率数），使模型能够自适应学习每个频率分量的重要性，而非依赖固定阈值。

Zhou, T., Ma, Z., Wen, Q., Wang, X., Sun, L., & Jin, R. (2022). FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting. In Proceedings of the 39th International Conference on Machine Learning (pp. 27268–27286). PMLR.

```python
# time_moe/models/experts/fedformer_token_expert.py
class FedFormerCycleExpert(BaseTokenExpert):
    def __init__(self, hidden_size, modes=32, ...):
        self.freq_weight = nn.Parameter(
            torch.randn(modes, hidden_size) * 0.02   # W_freq ∈ R^{m×H}，随机初始化
        )
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward_seq(self, x):    # x: [B, L, H]
        x_fft   = torch.fft.rfft(x, dim=1)            # X_f = FFT(X)
        m       = min(self.modes, x_fft.shape[1])
        out_fft = torch.zeros_like(x_fft)
        out_fft[:, :m, :] = x_fft[:, :m, :] * self.freq_weight[:m]  # 逐频率加权
        out = torch.fft.irfft(out_fft, n=x.shape[1], dim=1)         # IFFT 还原
        return self.norm(self.out_proj(out))
```

##### E7 异常专家：AnomalyTransformer

将经典自注意力与高斯距离先验混合，建模"正常关联模式"：

$$\text{Prior}_{ij} = \mathcal{N}(|i-j|;\ 0,\ \sigma_i^2),\quad \text{Series}_{ij} = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)_{ij}$$

$$\text{Attn} = 0.5 \cdot \text{Series} + 0.5 \cdot \text{Prior},\quad \mathbf{out} = \mathbf{W}_\text{proj} \cdot (\text{Attn} \cdot V)$$

其中 $Q, K, V \in \mathbb{R}^{L \times d}$ 分别为查询、键、值矩阵（$d$ 为注意力头维度）；$i, j$ 为序列位置索引；$\text{Prior}_{ij}$ 为位置 $i$ 对 $j$ 的高斯先验权重，$\sigma_i$ 由网络从当前 token 动态预测（自适应邻域宽度）；$\text{Attn}$ 为两类注意力的等权混合。正常时间点受高斯先验约束，只与邻近位置强相关；异常时间点则因跳离先验而在混合注意力分布中产生偏差。

```python
# time_moe/models/experts/anomaly_token_expert.py
class AnomalyTokenExpert(BaseTokenExpert):
    def forward_seq(self, x):   # x: [B, L, H]
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        sigma   = self.sigma_proj(x).transpose(1, 2)   # [B, heads, L]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        series = F.softmax(scores, dim=-1)             # 序列注意力
        prior  = self._gaussian_prior(sigma, length=L) # 高斯先验注意力

        mixed  = 0.5 * series + 0.5 * prior            # 等权混合
        out    = torch.matmul(mixed, v)
        return self.norm(self.out_proj(out.reshape(B, L, H)))
```

Xu, J., Wu, H., Wang, J., & Long, M. (2022). Anomaly transformer: Anomaly detection in time series with association discrepancy. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (pp. 2174–2184). ACM.

### 2. 改进门控模块

#### 2.1 原始流程

**Step 1：线性打分**

将 token 隐状态 $\mathbf{x} \in \mathbb{R}^H$ 经过 gate 线性层得到 logit：

$$\mathbf{l} = \mathbf{W}_g \mathbf{x}, \quad \mathbf{l} \in \mathbb{R}^N$$

其中 $N$ 为专家总数（Time-MoE 默认 $N=8$）。

**Step 2：Softmax 归一化**

$$p_i = \frac{e^{l_i}}{\sum_{j=1}^{N} e^{l_j}}, \quad i = 1, \ldots, N$$

其中 $p_i \in (0,1)$ 为路由器对第 $i$ 个专家的归一化路由概率，满足 $\sum_i p_i = 1$。

**Step 3：直接 Top-k 选取**

在全体专家概率上无约束地选取概率最大的 $k$ 个：

$$S = \text{TopK}(\mathbf{p},\ k)$$

其中 $S \subseteq \{1, \ldots, N\}$ 为被激活的专家下标集合，$|S|=k$（默认 $k=2$）。

**Step 4：稀疏融合**

$$\mathbf{h} = \sum_{i \in S} p_i \cdot f_i(\mathbf{x}) + \text{shared}(\mathbf{x})$$

其中 $f_i(\mathbf{x})$ 为第 $i$ 个专家的前向输出，$\text{shared}(\mathbf{x})$ 为共享专家的无条件激活输出（实际实现中经 sigmoid 标量门控缩放后叠加）。

#### 2.2 可能存在的问题

##### 1.专家同质化

见 1.2

##### 2.多样性无保证

当 $k=2$、专家池包含多种类型时，Top-k 可能从同一类型中选出全部 2 个专家。例如两个 trend 专家同时被选中，则该 token 的输出完全由趋势特征主导，丢失了周期和异常成分的信息：

这与 1.2 的目标不符。

##### 3.归纳偏置

不同类型的专家具有完全不同的归纳偏置：N-BEATS 的多项式基函数展开适合低频趋势，FEDformer 的频域变换适合周期建模，AnomalyTransformer 的距离先验适合异常检测。若路由不保证每次选中覆盖多种类型，领域特化的归纳偏置将被随机的门控竞争抵消，专家设计的初衷无法在训练中得到强化。

#### 2.3 改进流程

**Step 1：线性打分**

将 token 隐状态 $\mathbf{x} \in \mathbb{R}^H$ 经过 gate 线性层得到 logit：

$$\mathbf{l} = \mathbf{W}_g \mathbf{x}, \quad \mathbf{l} \in \mathbb{R}^N$$

其中 $N$ 为专家总数（Time-MoE 默认 $N=8$）。

**Step 2：Softmax 归一化**

$$p_i = \frac{e^{l_i}}{\sum_{j=1}^{N} e^{l_j}}, \quad i = 1, \ldots, N$$

其中 $p_i$ 为路由器的归一化路由概率。

**Step 3（新增）：类型内预选（Typed Preselection）**

对每种专家类型 $c \in \{\text{trend},\ \text{cycle},\ \text{anomaly}\}$，仅保留该类型内概率最大的候选者，其余同类专家概率置零：

$$\text{winner}_c = \arg\max_{j \in \mathcal{E}_c}\ p_j$$

$$\tilde{p}_{j} = \begin{cases} p_{j} & \text{if}\ j = \text{winner}_c \text{ for some } c \\ 0 & \text{otherwise} \end{cases}$$
其中 $\mathcal{E}_c$ 为类型 $c$ 对应的专家下标集合，$\text{winner}_c$ 为该类型内路由概率最高的候选专家；$\tilde{p}_j$ 为过滤后的稀疏概率向量，非候选项置零。
经过此步骤，候选集 $\tilde{\mathbf{p}}$ 中至多只有 3 个非零值（每类一个候选）。

**Step 4：跨类型 Top-k**

在过滤后的候选集上进行 Top-k 选取：

$$S = \text{TopK}(\tilde{\mathbf{p}},\ k)$$

其中 $\tilde{\mathbf{p}}$ 为类型内预选后的稀疏概率向量（至多 $C=3$ 个非零项）；当 $k=2$、类型数 $C=3$ 时，从 3 个候选中选出 2 个，被选中的专家**必然来自不同类型**。

**Step 5：稀疏融合**

$$\mathbf{h} = \sum_{i \in S} \tilde{p}_i \cdot f_i(\mathbf{x}) + \text{shared}(\mathbf{x})$$

其中权重 $\tilde{p}_i$ 来自类型预选后的稀疏概率分布，保证跨类专家的贡献均以正値权重混合。

```python
# time_moe/models/typed_router_utils.py
def typed_preselect(routing_weights, expert_type_ids):
    """类型内预选：每种类型记留得分最高的一个候选"""
    filtered = torch.zeros_like(routing_weights)   # [T, N] 全零初始化
    num_types = int(expert_type_ids.max()) + 1
    for type_id in range(num_types):               # 每种类型独立处理
        type_mask  = (expert_type_ids == type_id)
        type_probs = routing_weights[:, type_mask]  # 取出该类型所有专家的概率
        best_local = type_probs.argmax(dim=-1)      # argmax winner_c
        global_idx = type_mask.nonzero()[best_local]
        # 只将 winner 的概率写入 filtered
        filtered[token_indices, global_idx] = routing_weights[token_indices, global_idx]
    return filtered  # 至多 num_types 个非零元素
```

### 3. 设计冻结训练 

背景：基于预训练的 `Time-MoE-50M` 继续训练：

#### 3.1 原因

**1. 灾难性遗忘**

5 个新引入的异构专家（E1/E2/E4/E5/E7）在初始阶段尚未经过任何训练，其输出是随机噪声。这些随机输出通过路由权重混入隐状态，会产生巨大的梯度信号回传到预训练主干（Self-Attention、Embedding 等），导致原本已高质量收敛的预训练权重被快速覆写，性能出现断崖式退化。

**2. 路由与专家的初始耦合死锁**

Gate 网络和专家网络在优化上高度相互依赖：Gate 需要根据专家的真实能力决定分发对象；专家需要接收到符合其归纳偏置的 token 分布后才能逐步习得对应特征。在两者均处于随机初始状态时同步训练，极易陷入"垃圾进垃圾出"的恶性循环，收敛到低质量局部极小。

Lepikhin et al.（2021, GShard）指出 MoE 训练中路由与专家的联合优化存在固有不稳定性，并强调初始阶段专家分配的均衡对收敛至关重要，支持先稳定路由后激活专家的顺序设计。

Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y., Krikun, M., Shazeer, N., & Chen, Z. (2021). GShard: Scaling giant models with conditional computation and automatic sharding. In *Proceedings of the 9th International Conference on Learning Representations* (ICLR). https://arxiv.org/abs/2006.16668

#### 3.2 解决方法

实现 `PhasedFreezeCallback`，在训练过程中动态控制参数的可训练性，将训练过程划分为三个阶段：

$$\text{Phase-A} \xrightarrow{\text{step}=T_A} \text{Phase-B} \xrightarrow{\text{step}=T_B} \text{Phase-C}$$

其中 $T_A$（训练配置项 `phase_a_end`）与 $T_B$（`phase_b_end`）为阶段切换步数，可在 YAML 配置文件中灵活调整。

```python
# time_moe/trainer/hf_trainer.py
class PhasedFreezeCallback(TrainerCallback):
    def __init__(self, model, config, phase_a_end=1000, phase_b_end=5000):
        self.phase_a_end = phase_a_end     # T_A
        self.phase_b_end = phase_b_end     # T_B
        # 新增异构专家的名称模式列表
        new_idx = _identify_new_expert_indices(config)  # zero_init_output=true 的专家编号
        self._gate_patterns       = [".gate."]                              # Phase-A
        self._new_expert_patterns = [f".experts.{i}." for i in new_idx]    # Phase-B
        self._apply_phase_a(model)  # 立即进入 Phase-A

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if   step < self.phase_a_end: self._apply_phase_a(model)  # 只训 gate
        elif step < self.phase_b_end: self._apply_phase_b(model)  # gate + 新专家
        else:                         self._apply_phase_c(model)  # 全参数
```

| 阶段                      | 步数范围          | 可训练参数                                | 冻结参数             |
| ------------------------- | ----------------- | ----------------------------------------- | -------------------- |
| **Phase-A**（Gate 适配）  | $[0,\ T_A)$       | 仅 gate 线性层（`ffn_layer.gate.weight`） | 主干 + 所有专家      |
| **Phase-B**（新专家预热） | $[T_A,\ T_B)$     | gate + 新异构专家（E1/E2/E4/E5/E7）       | 主干 + 原始 MLP 专家 |
| **Phase-C**（全参数联调） | $[T_B,\ +\infty)$ | 全部参数                                  | 无                   |

Howard, J., & Ruder, S. (2018). Universal language model fine-tuning for text classification. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics* (ACL), Vol. 1, 328–339. https://doi.org/10.18653/v1/P18-1031

分阶段冻结与零初始化残差在两个不同层面共同保护预训练知识，形成双重防护：

```
零初始化残差（数值层面）
  └─ 新专家输出 ≡ 0，隐状态数值上不受干扰

分阶段冻结（梯度层面）
  └─ Phase-A：主干梯度被阻断，反向传播不更新预训练权重
  └─ Phase-B：主干仍冻结，新专家在有限范围内安全学习
```

两者的保护范围互补：即使某次实验取消零初始化（如复用预训练 MLP 的 E0/E3/E6），分阶段冻结仍然保护主干；即使训练步数极短导致阶段切换不够平稳，零初始化也保证第一步的数值稳定性。

这与 LoRA 的设计一致：新增模块在初始时刻对模型输出贡献为零，通过渐进训练实现向原始能力的平滑叠加。

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. In *Proceedings of the 10th International Conference on Learning Representations* (ICLR). https://arxiv.org/abs/2106.09685

### 3.3 归纳偏置设计

 在三个层次上同时施加领域先验：

| 层次                       | 机制                                | 对应先验                               | 激活时机                            |
| -------------------------- | ----------------------------------- | -------------------------------------- | ----------------------------------- |
| **结构层**（专家算子拓扑） | 移动平均、FFT、高斯注意力的网络拓扑 | $Y_t = T_t + S_t + R_t$ 三分量独立存在 | $t=0$ 即存在，不依赖训练            |
| **路由层**（Gate 约束）    | `typed_preselect` 强制跨类型选择    | 每个 token 的表示包含多种时序成分      | Phase-A 开始，Gate 学习期间逐步强化 |
| **初始化层**（零初始化）   | 新专家训练起点为零贡献              | 预训练知识优先，新知识渐进融入         | $t=0$ 保护，Phase-B 后渐进激活      |

结构层的归纳偏置内嵌于算子拓扑，在参数初始化前即已存在，不依赖任何训练数据；路由层约束在 Phase-A 开始就通过 Gate 的梯度更新逐步体现；结构层的能力则在 Phase-B 之后随专家权重的实质性更新才真正发挥作用。

---

### 4. 初步实验

#### 4.1 实验环境

| 项目 | 规格 |
| ---- | ---- |
| GPU | NVIDIA GeForce RTX 5060（8 GB VRAM） |
| CUDA | 12.8 |
| PyTorch | 2.11.0.dev20260130+cu128 |
| Transformers | 4.40.1 |
| Python | 3.10.19（conda 环境） |
| 操作系统 | Windows 11（单卡，eager attention） |

#### 4.2 训练配置

以 ETTh1 数据集（前 60%，约 10452 个时间点/列）为训练集，进行端到端测试。核心超参：

| 参数 | 值 |
| ---- | -- |
| 数据集 | ETTh1（7 特征列） |
| 上下文窗口 | 128 |
| 训练步数 | 100 |
| 全局批大小 | 4（梯度累积 ×2） |
| 峰值学习率 | $5 \times 10^{-4}$（cosine 衰减至 $10^{-5}$） |
| 精度 | bf16 |
| Phase-A 结束步 $T_A$ | 20 |
| Phase-B 结束步 $T_B$ | 50 |

训练损失与梯度范数变化：

| 步骤 | 训练损失 | 梯度范数 | 阶段 |
| ---- | -------- | -------- | ---- |
| 10 | 0.1524 | 0.058 | Phase-A |
| 20 | 0.1920 | 0.053 | Phase-A → B |
| 50 | 0.1622 | 0.404 | Phase-B → C |
| 80 | 0.1913 | 7.563 | Phase-C |
| 100 | 0.1518 | 2.297 | Phase-C |

Phase-A 阶段（仅 gate 可训练）梯度范数极小（$\approx 0.05$），主干参数完全受保护；进入 Phase-C 后梯度范数骤升至 7.5，反映全量参数开始适配异构专家，训练尚未收敛。

#### 4.3 评估结果

测试集为 ETTh1 后 20%，上下文长度 512，预测长度 96。

| 模型 | MSE | MAE |
| ---- | --- | --- |
| TimeMoE-50M（原始，0 步） | 0.357681 | 0.381649 |
| Type-MoE（100 步，ctx=128 训练） | 0.362939 | 0.390024 |
| Type-MoE（100 步，ctx=512 训练） | 0.356523 | 0.386796 |
| **Type-MoE（300 步，ctx=512，ETTh1+Electricity）** | **0.355036** | **0.386348** |
| 差值（最终 vs 原始） | -0.0026 | +0.0047 |
| 相对差异 | -0.74% | +1.23% |

对齐训练与评估上下文（均为 512）后，MSE 首次低于原始基线。进一步引入 Electricity 数据（8 列均匀采样，15782 点 / 列）并将训练步数增加至 300 步后，MSE 进一步降至 0.3550，小幅益续提升。训练数据多样性的提升与训练步数的增加共同贡献了这一改善。

#### 4.4 路由多样性验证

对 12 层的路由分配进行统计，三类专家的全局选择比例如下：

| 专家类型 | 代表专家 | 全局路由比例 |
| -------- | -------- | ------------ |
| trend | E0 MLP / E1 N-BEATS / E2 Autoformer-trend | **36.1%** |
| cycle | E3 MLP / E4 Autoformer-cycle / E5 FEDformer | **37.5%** |
| anomaly | E6 MLP / E7 AnomalyTransformer | **26.4%** |

三类专家路由比例相对均衡，无路由坍缩现象，`typed_topk` 路由成功强制保证了跨类型多样性。各层的专家偏好亦呈现明显分化：浅层（Layer 0–1）倾向 N-BEATS 与 MLP 等轻量趋势/周期专家，深层（Layer 10–11）更多调用 Autoformer-trend，符合"浅层特征提取、深层模式建模"的预期。

---

### 5. 后续计划

目前实验仅为 100 步冒烟测试，距完整调优仍有较大差距，后续将从以下方向推进：

**1. 延长训练步数**

Phase-C 阶段的梯度范数仍在 2–8 之间波动，表明模型远未收敛。计划以 2000 步（$T_A=200, T_B=800$）进行标准调优，预期 MSE 可在基线基础上下降 10–15%。

**2. 对齐训练与评估上下文**

将训练上下文从 128 扩展至 512，消除训练-推理分布偏移，进一步挖掘 seq 接口专家（E2/E4/E5/E7）在长序列建模上的潜力。

**3. 扩展训练数据**

引入 ETTh2 / ETTm1 / ETTm2 / Weather / Electricity 等多数据集，为趋势、周期、异常三类专家提供更充足且多样的训练样本，促进专家分工的自发涌现。

**4. 标准长度评估**

在完整训练后，对 pred\_len ∈ {96, 192, 336, 720} 全套预测长度进行评估，与原始 TimeMoE-50M 及 PatchTST 等基线进行系统对比。
