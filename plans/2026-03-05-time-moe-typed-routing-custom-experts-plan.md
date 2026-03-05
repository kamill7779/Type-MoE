# Type-MoE：类型约束路由 + 异构专家实施计划（2026-03-05）

> **目标**：将 Time-MoE 改造为 Type-MoE —— 在门控路由中引入类型约束（typed top-k），并将同构 MLP 专家有计划地替换为 Autoformer / FEDformer / N-BEATS / AnomalyTransformer 风格的异构专家。
>
> **核心决策**：seq 型专家采用 **S1 方案（全序列执行 + 选择性聚合）**，保证算子语义完整性。
>
> **约束**：本文件为完整实施方案，不含代码实现。

---

## 1. 决策冻结

以下决策在本计划执行期间不可更改：

| # | 决策项 | 结论 |
|---|-------|------|
| 1 | 路由粒度 | **token 级**（严格 Time-MoE 风格） |
| 2 | 路由策略 | `softmax → jitter noise → 类型内预选(每类保留1个) → top-k` |
| 3 | 默认 top_k | 2 |
| 4 | norm_topk_prob | false |
| 5 | 共享专家 | 保留常驻分支，不参与路由 |
| 6 | 专家类型 | trend / cycle / anomaly，每类型 ≥ 2 个专家 |
| 7 | seq 专家调度 | **S1：全序列执行 + 选择性聚合** |
| 8 | 训练目标 | 保持 Time-MoE AR 训练 |
| 9 | 专家总数 | 保持与预训练 checkpoint 一致（默认 8），确保 gate 权重可复用 |

---

## 2. 现状基线（Time-MoE 关键实现）

### 2.1 代码结构

| 组件 | 类名 | 位置 |
|------|------|------|
| 输入嵌入 | `TimeMoeInputEmbedding` | `modeling_time_moe.py` ~L172 |
| 稀疏专家层 | `TimeMoeSparseExpertsLayer` | `modeling_time_moe.py` ~L265 |
| 单个专家 | `TimeMoeTemporalBlock` | `modeling_time_moe.py` ~L248 |
| Decoder 层 | `TimeMoeDecoderLayer` | `modeling_time_moe.py` ~L659 |
| 模型主体 | `TimeMoeModel` | `modeling_time_moe.py` ~L780 |
| 预测头+损失 | `TimeMoeForPrediction` | `modeling_time_moe.py` ~L939 |
| 配置 | `TimeMoeConfig` | `configuration_time_moe.py` |
| 训练入口 | `TimeMoeRunner` | `runner.py` |

### 2.2 当前路由流程

```
hidden_states: [B, L, H]
  → view(-1, H)                          # [B*L, H]
  → gate(x)                              # router_logits: [B*L, N_experts]
  → softmax                              # routing_weights: [B*L, N_experts]
  → topk(k=2)                            # weights + indices: [B*L, 2]
  → one_hot → permute                    # expert_mask: [N_experts, 2, B*L]
  → per-expert loop:
      hidden[top_x] → expert(·) × weight → index_add_
  → + shared_expert(x) × sigmoid(gate)
  → view(B, L, H)
```

### 2.3 当前 aux loss

标准 Switch Transformer 负载均衡损失：`num_experts × Σ(f_i × P_i)`，其中 `f_i` 为 token 分配比例，`P_i` 为路由概率均值。

---

## 3. 目标架构

### 3.1 主链路

```
x → TimeMoeInputEmbedding
  → N × DecoderLayer(
      RMSNorm → SelfAttn → Residual
      → RMSNorm → TypedSparseExpertsLayer → Residual
    )
  → final RMSNorm → AR head
```

### 3.2 TypedSparseExpertsLayer 路由流程

```
hidden_states: [B, L, H]
  → view(-1, H)                                      # x_flat: [B*L, H]
  → gate(x_flat)                                     # router_logits: [B*L, N]
  → (training) logits += randn × jitter_noise        # 探索噪声
  → softmax                                          # routing_weights: [B*L, N]
  → typed_preselect(weights, type_map)               # filtered: [B*L, N] 每类型仅保留 argmax
  → topk(k=2)                                        # topk_weights, topk_indices: [B*L, k]
  → 分流 dispatch:
      flat expert: x_flat[top_x] → expert(·)         # [M_i, H] → [M_i, H]
      seq expert:  hidden_states → expert(·)          # [B, L, H] → [B, L, H], 取 [top_x] 位置
      → × topk_weight → index_add_
  → + shared_expert(x_flat) × sigmoid(shared_gate)
  → view(B, L, H)
```

### 3.3 S1 调度策略详解

seq 型专家（Autoformer / FEDformer / AnomalyTransformer）需要完整序列上下文（FFT、移动平均、序列注意力无法在离散 token 子集上执行）。

**S1 方案：全序列执行 + 选择性聚合**

```
对于 seq 型专家 expert_j:
1. 始终在完整 [B, L, H] 上执行 → seq_out: [B, L, H]
2. 将 seq_out 展平为 [B*L, H]
3. 仅取 router 选中该专家的 token 位置 top_x
4. selected = seq_out_flat[top_x] × topk_weight
5. index_add_ 回 final_hidden

未选中位置的计算结果被丢弃。
```

**权衡**：
- 优点：算子语义完整，实现简单，正确性有保证
- 缺点：seq 专家计算量不随选中 token 数减少
- 可接受性：每层 seq 专家仅 3-4 个，且 top-k=2 意味着大部分 token 不选 seq 专家时，其输出被丢弃但计算仍执行。实测开销增量约 30%~80%（见 §12）

---

## 4. 路由算法规范

### 4.1 类型约束 Top-k

对每个 token（展平后索引 t）：

1. `logits_t = gate(x_t)` → `[N_experts]`
2. （训练时）`logits_t += randn × jitter_noise`（`jitter_noise ∈ [0.01, 0.1]`）
3. `p_t = softmax(logits_t)` → `[N_experts]`
4. 按 `expert_type_map` 分组（trend=0 / cycle=1 / anomaly=2）
5. 每组仅保留组内 argmax 专家概率，其余置 0 → `filtered_t`
6. `topk(filtered_t, k=2)` → `topk_weights, topk_indices`
7. 对选中专家的输出按 `topk_weights` 加权聚合

### 4.2 `typed_preselect` 函数

```python
def typed_preselect(
    routing_weights: torch.Tensor,   # [T, N_experts]
    expert_type_ids: torch.Tensor,   # [N_experts], 值域 [0, num_types)
    num_types: int,
) -> torch.Tensor:
    """每个 token，每个类型仅保留该类型内概率最大的专家，其余置 0。"""
    filtered = torch.zeros_like(routing_weights)
    for type_id in range(num_types):
        type_mask = (expert_type_ids == type_id)                    # [N_experts]
        type_probs = routing_weights[:, type_mask]                  # [T, n_in_type]
        best_in_type = type_probs.argmax(dim=-1)                    # [T]
        global_indices = type_mask.nonzero(as_tuple=True)[0]        # [n_in_type]
        best_global = global_indices[best_in_type]                  # [T]
        token_indices = torch.arange(routing_weights.size(0), device=routing_weights.device)
        filtered[token_indices, best_global] = routing_weights[token_indices, best_global]
    return filtered
```

> **优化提示**：P1 用 for-loop 保证正确性；P4 可通过 scatter/gather 向量化。

### 4.3 Aux Loss 设计

**关键决策**：aux loss 中的 `P_i` 使用**预选前的原始 softmax 概率**（而非 filtered_probs），确保所有专家都能收到路由梯度，防止组内马太效应。

```
aux_loss = num_experts × Σ(f_i × P_i) × aux_factor
```

- `f_i`：由 `topk_indices`（预选后 top-k 结果）统计的 token 分配比例
- `P_i`：由**原始** `routing_weights`（预选前 softmax 概率）统计的平均路由概率
- `actual_k = min(config.top_k, 有效候选数)`

### 4.4 类型多样性损失

附加损失项，鼓励每个 token 的 top-k 覆盖不同类型：

```python
def type_diversity_loss(topk_indices, expert_type_ids, num_types, top_k):
    """鼓励 top-k 选择覆盖不同类型。top_k >= 2 时生效。"""
    selected_types = expert_type_ids[topk_indices]                  # [B*L, top_k]
    unique_types = torch.zeros(topk_indices.size(0), num_types, device=topk_indices.device)
    unique_types.scatter_(1, selected_types, 1.0)
    num_unique = unique_types.sum(dim=-1)                           # [B*L]
    target = min(top_k, num_types)
    return (target - num_unique).clamp(min=0).mean()
```

总损失 = AR loss + `aux_factor × aux_loss` + `type_diversity_factor × diversity_loss`

### 4.5 路由元信息输出

每层记录（用于训练监控与离线分析）：

| 字段 | shape | 说明 |
|------|-------|------|
| `router_logits` | `[B*L, N]` | 原始 gate 输出 |
| `topk_indices` | `[B*L, k]` | 选中的专家索引 |
| `topk_weights` | `[B*L, k]` | 选中的路由权重 |
| `filtered_probs` | `[B*L, N]`（可选） | 类型预选后概率，便于诊断 |

---

## 5. 专家体系设计

### 5.1 统一专家基类

```python
class BaseTokenExpert(nn.Module):
    """所有 Type-MoE 专家的基类"""
    expert_type: str = "generic"       # "trend" | "cycle" | "anomaly"
    interface_kind: str = "flat"       # "flat" | "seq"

    def forward_flat(self, x: torch.Tensor) -> torch.Tensor:
        """[M, H] → [M, H]，稀疏友好"""
        raise NotImplementedError

    def forward_seq(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, H] → [B, L, H]，序列算子"""
        raise NotImplementedError

    def forward(self, x, **kwargs):
        if self.interface_kind == "flat":
            return self.forward_flat(x)
        else:
            return self.forward_seq(x)
```

**强制约定**：所有专家输出前必须经过 RMSNorm，保证异构专家输出尺度对齐。

### 5.2 专家类型池与默认布局

默认 8 专家（兼容预训练 checkpoint），布局建议：

| 索引 | 专家名 | 类型 | 接口 | 来源 |
|------|--------|------|------|------|
| 0 | MLP-Trend-0 | trend | flat | 原 TimeMoeTemporalBlock（复用预训练权重） |
| 1 | NBeats-Trend | trend | flat | N-BEATS block 重构 |
| 2 | Autoformer-Trend | trend | seq | Autoformer series decomp → trend |
| 3 | MLP-Cycle-0 | cycle | flat | 原 TimeMoeTemporalBlock（复用预训练权重） |
| 4 | Autoformer-Cycle | cycle | seq | Autoformer series decomp → seasonal + auto-correlation |
| 5 | FedFormer-Cycle | cycle | seq | FEDformer Fourier enhanced block |
| 6 | MLP-Anomaly-0 | anomaly | flat | 原 TimeMoeTemporalBlock（复用预训练权重） |
| 7 | Anomaly-Attn | anomaly | seq | AnomalyTransformer anomaly attention |

对应 `expert_type_map = [0, 0, 0, 1, 1, 1, 2, 2]`（trend=0, cycle=1, anomaly=2）

> **设计原则**：每个类型至少保留 1 个原始 MLP 专家作为基线/后备，确保组内有竞争、aux loss 均衡。

### 5.3 Dispatch 伪代码（完整版）

```python
def forward(self, hidden_states: torch.Tensor):
    B, L, H = hidden_states.shape
    x_flat = hidden_states.view(-1, H)                              # [B*L, H]

    # --- 路由 ---
    router_logits = self.gate(x_flat)                               # [B*L, N]
    if self.training and self.jitter_noise > 0:
        router_logits = router_logits + torch.randn_like(router_logits) * self.jitter_noise

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

    # --- 类型约束预选 ---
    filtered_weights = typed_preselect(routing_weights, self.expert_type_ids, self.num_types)

    # --- Top-k ---
    topk_weights, topk_indices = torch.topk(filtered_weights, self.top_k, dim=-1)
    if self.norm_topk_prob:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(hidden_states.dtype)

    # --- Expert dispatch ---
    final_hidden = torch.zeros_like(x_flat)
    expert_mask = F.one_hot(topk_indices, self.num_experts).permute(2, 1, 0)

    # 预计算 seq 专家输出（S1：全序列执行，仅在有 token 命中时执行）
    seq_expert_cache = {}
    for expert_idx in range(self.num_experts):
        if self.experts[expert_idx].interface_kind == "seq":
            if expert_mask[expert_idx].any():
                seq_expert_cache[expert_idx] = self.experts[expert_idx].forward_seq(
                    hidden_states
                ).view(-1, H)

    # 分发聚合
    for expert_idx in range(self.num_experts):
        expert = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx])

        if len(top_x) == 0:
            continue

        if expert.interface_kind == "flat":
            current = x_flat[top_x]
            out = expert.forward_flat(current) * topk_weights[top_x, idx, None]
            final_hidden.index_add_(0, top_x, out.to(x_flat.dtype))

        elif expert.interface_kind == "seq":
            seq_out_flat = seq_expert_cache[expert_idx]
            selected = seq_out_flat[top_x] * topk_weights[top_x, idx, None]
            final_hidden.index_add_(0, top_x, selected.to(x_flat.dtype))

    # --- 共享专家（常驻） ---
    shared_out = self.shared_expert(x_flat)
    shared_gate = F.sigmoid(self.shared_expert_gate(x_flat))
    final_hidden = final_hidden + shared_gate * shared_out

    return final_hidden.view(B, L, H), router_logits
```

---

## 6. 各外部算子的集成规范

### 6.1 原则

- **不**整模型搬运，只做算子级抽取 + wrapper 封装
- 每个专家内部自行管理数值精度（如 FFT upcast float32）
- 输出统一经过 RMSNorm 对齐尺度

### 6.2 Autoformer 专家（trend + cycle）

**提取算子**：

1. **SeriesDecomposition**：移动平均 → trend + seasonal 分离
   - 输入 `[B, L, H]`，参数 `kernel_size`，输出 `(trend, seasonal)` 各 `[B, L, H]`

2. **SimplifiedAutoCorrelation**：FFT 自相关 → top-k 频率选择 → 周期对齐聚合
   - 输入 `[B, L, H]`，输出 `[B, L, H]`
   - P3 先实现简化版（top-k 频率 + 对齐），不做完整 time-delay aggregation

```python
class AutoformerTrendExpert(BaseTokenExpert):
    expert_type = "trend"
    interface_kind = "seq"

    def __init__(self, hidden_size, kernel_size=25):
        super().__init__()
        self.decomp = SeriesDecomposition(kernel_size)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = RMSNorm(hidden_size)

    def forward_seq(self, x):                   # [B, L, H]
        trend, _ = self.decomp(x)
        return self.norm(self.proj(trend))


class AutoformerCycleExpert(BaseTokenExpert):
    expert_type = "cycle"
    interface_kind = "seq"

    def __init__(self, hidden_size, kernel_size=25, top_k_freq=3):
        super().__init__()
        self.decomp = SeriesDecomposition(kernel_size)
        self.auto_corr = SimplifiedAutoCorrelation(hidden_size, top_k_freq)
        self.norm = RMSNorm(hidden_size)

    def forward_seq(self, x):
        _, seasonal = self.decomp(x)
        return self.norm(self.auto_corr(seasonal))
```

### 6.3 FEDformer 专家（cycle）

**提取算子**：FourierEnhancedBlock

- `rfft → 低频截断(modes 个频率) → 可学习频域线性变换 → irfft`
- 核心参数：`modes`（保留频率数，默认 32）

```python
class FedFormerCycleExpert(BaseTokenExpert):
    expert_type = "cycle"
    interface_kind = "seq"

    def __init__(self, hidden_size, modes=32):
        super().__init__()
        self.modes = modes
        self.freq_weight = nn.Parameter(torch.randn(modes, hidden_size) * 0.02)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = RMSNorm(hidden_size)

    def forward_seq(self, x):                   # [B, L, H]
        # upcast for FFT numerical stability
        x_fp32 = x.float()
        x_fft = torch.fft.rfft(x_fp32, dim=1)                          # [B, L//2+1, H]
        x_fft_trunc = x_fft[:, :self.modes, :]                         # [B, modes, H]
        x_fft_trunc = x_fft_trunc * self.freq_weight.float().unsqueeze(0)
        out_fft = torch.zeros_like(x_fft)
        out_fft[:, :self.modes, :] = x_fft_trunc
        out = torch.fft.irfft(out_fft, n=x.size(1), dim=1)            # [B, L, H]
        return self.norm(self.proj(out.to(x.dtype)))
```

> **注意**：FFT 在 bf16 下精度差，专家内部 upcast float32 执行 FFT 再 downcast。

### 6.4 AnomalyTransformer 专家（anomaly）

**提取算子**：简化版 AnomalyAttention

- Prior-association：可学习 sigma 高斯核先验（越近越可能关联）
- Series-association：标准自注意力
- 输出 = series attention 加权的 value

```python
class AnomalyTokenExpert(BaseTokenExpert):
    expert_type = "anomaly"
    interface_kind = "seq"

    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.sigma = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = RMSNorm(hidden_size)

    def forward_seq(self, x):                   # [B, L, H]
        B, L, H = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]

        # Series association
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)

        # Prior association (Gaussian distance kernel) — 供未来 association discrepancy 使用
        # dist = |i - j|, prior = exp(-0.5 * (dist / sigma)^2), row-normalized
        # 当前版本仅用 series attention 输出

        out = (attn @ v).transpose(1, 2).reshape(B, L, H)
        return self.norm(self.out_proj(out))
```

### 6.5 N-BEATS 专家（trend, flat）

**提取思想**：FC stack → theta 参数 → 基函数展开（多项式基表示趋势）

```python
class NBeatsTokenExpert(BaseTokenExpert):
    expert_type = "trend"
    interface_kind = "flat"

    def __init__(self, hidden_size, num_layers=4, theta_dim=8):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden_size, hidden_size, bias=False), nn.ReLU()])
        self.stack = nn.Sequential(*layers)
        self.theta_proj = nn.Linear(hidden_size, theta_dim, bias=False)
        self.basis_proj = nn.Linear(theta_dim, hidden_size, bias=False)
        self.norm = RMSNorm(hidden_size)

    def forward_flat(self, x):                  # [M, H]
        h = self.stack(x)
        theta = self.theta_proj(h)
        return self.norm(self.basis_proj(theta))
```

---

## 7. 专家注册表（Plugin Registry）

### 7.1 Registry 设计

```python
# time_moe/models/experts/registry.py

EXPERT_REGISTRY: Dict[str, Type[BaseTokenExpert]] = {}

def register_expert(name: str):
    def decorator(cls):
        EXPERT_REGISTRY[name] = cls
        return cls
    return decorator

def build_expert(spec: dict, hidden_size: int) -> BaseTokenExpert:
    """从配置 dict 实例化专家。"""
    cls = EXPERT_REGISTRY[spec["name"]]
    params = spec.get("params", {})
    return cls(hidden_size=hidden_size, **params)

def build_expert_list(specs: List[dict], hidden_size: int) -> nn.ModuleList:
    """从配置列表构建完整专家 ModuleList。"""
    return nn.ModuleList([build_expert(s, hidden_size) for s in specs])
```

### 7.2 每个专家注册

```python
@register_expert("mlp_temporal_block")
class MLPTemporalExpert(BaseTokenExpert): ...

@register_expert("nbeats_trend")
class NBeatsTokenExpert(BaseTokenExpert): ...

@register_expert("autoformer_trend")
class AutoformerTrendExpert(BaseTokenExpert): ...

@register_expert("autoformer_cycle")
class AutoformerCycleExpert(BaseTokenExpert): ...

@register_expert("fedformer_cycle")
class FedFormerCycleExpert(BaseTokenExpert): ...

@register_expert("anomaly_attn")
class AnomalyTokenExpert(BaseTokenExpert): ...
```

### 7.3 State Dict 兼容性

- `self.experts[i]` 的 state_dict key 保持为 `experts.{i}.xxx`
- 原 `TimeMoeTemporalBlock` 封装为 `MLPTemporalExpert`，内部结构不变，权重 key 兼容
- 新型专家无预训练权重，使用 zero-init residual（见 §9）

---

## 8. 配置变更

### 8.1 新增字段（`configuration_time_moe.py`）

```python
# === Type-MoE 新增 ===
router_mode: str = "standard"           # "standard" | "typed_topk"
expert_types: List[str] = []            # ["trend", "cycle", "anomaly"]
expert_type_map: List[int] = []         # [0, 0, 0, 1, 1, 1, 2, 2]  len=num_experts
norm_topk_prob: bool = False
jitter_noise: float = 0.05             # 路由探索噪声
type_diversity_factor: float = 0.01    # 类型多样性损失系数
seq_expert_mode: str = "full_seq"      # "full_seq" (S1) | "local_window" (S3, 未来)
seq_expert_window: int = 64            # S3 模式窗口大小（S1 下不使用）
expert_output_norm: bool = True        # 专家输出是否加 RMSNorm
custom_expert_specs: List[dict] = []   # 每个专家的详细配置
freeze_strategy: str = "none"          # "none" | "phased" | "gate_only"
```

### 8.2 配置模板（`configs/typed_experts/base.yaml`）

```yaml
model:
  hidden_size: 4096
  num_hidden_layers: 32
  num_attention_heads: 32
  num_experts: 8
  num_experts_per_tok: 2
  horizon_lengths: [1, 4, 12, 24]

router:
  router_mode: typed_topk
  expert_types: [trend, cycle, anomaly]
  expert_type_map: [0, 0, 0, 1, 1, 1, 2, 2]  # trend×3, cycle×3, anomaly×2
  norm_topk_prob: false
  jitter_noise: 0.05
  router_aux_loss_factor: 0.02
  type_diversity_factor: 0.01

experts:
  - {name: mlp_temporal_block, type: trend, interface: flat}
  - {name: nbeats_trend, type: trend, interface: flat, params: {num_layers: 4, theta_dim: 8}}
  - {name: autoformer_trend, type: trend, interface: seq, params: {kernel_size: 25}}
  - {name: mlp_temporal_block, type: cycle, interface: flat}
  - {name: autoformer_cycle, type: cycle, interface: seq, params: {kernel_size: 25, top_k_freq: 3}}
  - {name: fedformer_cycle, type: cycle, interface: seq, params: {modes: 32}}
  - {name: mlp_temporal_block, type: anomaly, interface: flat}
  - {name: anomaly_attn, type: anomaly, interface: seq, params: {num_heads: 8}}

training:
  freeze_strategy: phased
  precision: bf16
  learning_rate: 1e-5
  warmup_steps: 1000
```

---

## 9. 预训练权重迁移策略

从 `Salesforce/time-moe` 预训练 checkpoint（8 个同构 `TimeMoeTemporalBlock`）迁移至 8 个异构专家：

### 9.1 权重初始化规则

| 组件 | 初始化方式 | 说明 |
|------|-----------|------|
| Gate 权重 | **保留**原 checkpoint | 专家总数不变(8)，gate 输出维度兼容 |
| MLP-Trend-0 (idx=0) | **复制**原 expert[0] | 完全兼容 |
| MLP-Cycle-0 (idx=3) | **复制**原 expert[3] | 完全兼容 |
| MLP-Anomaly-0 (idx=6) | **复制**原 expert[6] | 完全兼容 |
| NBeats-Trend (idx=1) | Xavier init + **输出投影 zero-init** | zero-init residual 防初始阶段扰动 |
| Autoformer-Trend (idx=2) | Xavier init + **输出投影 zero-init** | 同上 |
| Autoformer-Cycle (idx=4) | Xavier init + **输出投影 zero-init** | 同上 |
| FedFormer-Cycle (idx=5) | Xavier init + **输出投影 zero-init** | 同上 |
| Anomaly-Attn (idx=7) | Xavier init + **输出投影 zero-init** | 同上 |
| Shared expert | **保留**原 checkpoint | 结构不变 |

### 9.2 Zero-Init Residual 技巧

新型专家的最后一层投影（`proj` / `out_proj` / `basis_proj`）权重初始化为 0。效果：训练初期新专家输出 ≈ 0，模型行为等价于仅使用原 MLP 专家 → 稳定过渡。

---

## 10. 训练策略

### 10.1 分阶段解冻协议

| 阶段 | 训练步 | 可训练参数 | 学习率 | 目的 |
|------|--------|-----------|--------|------|
| Phase-A: Gate Adaptation | 0 ~ 1,000 | gate 权重 only | 1e-4 | 让 gate 适应类型约束路由 |
| Phase-B: New Expert Warmup | 1,000 ~ 5,000 | 新专家 + gate | 5e-5 | 新专家从 zero-init 开始学习 |
| Phase-C: Full Joint | 5,000+ | 全部参数 | cosine → 1e-5 | 联合优化 |

### 10.2 损失函数

```
total_loss = AR_loss + aux_factor × aux_loss + diversity_factor × type_diversity_loss
```

- AR_loss：Huber loss（δ=2.0），多 horizon 平均
- aux_loss：Switch-style load balancing（`P_i` 用预选前概率）
- type_diversity_loss：鼓励 top-k 覆盖不同类型

### 10.3 梯度稳定化

- 每个专家输出加 RMSNorm（对齐异构专家输出尺度）
- FFT 型专家内部 upcast float32
- 可选：为异构专家设置独立 lr multiplier（optimizer param_groups 分组）
- `max_grad_norm = 1.0`（沿用原配置）

---

## 11. 推理适配

### 11.1 KV-Cache 模式下的 Seq 专家

**问题**：autoregressive 生成时每步仅 1 个新 token，seq 专家无法在单 token 上执行 FFT / 移动平均。

**解决方案**：Seq Expert Hidden Cache

```
每个 seq 专家维护独立的 hidden_states 缓存：
1. 首次前向：缓存完整 [B, L, H]
2. 后续步骤：将新 token hidden 追加到缓存 → [B, L+1, H]
3. seq 专家在完整缓存上执行
4. 只取最后一个位置的输出
```

**备选方案**：推理时将 seq 专家降级为 flat 近似（用 Phase-C 阶段学到的统计量）。

### 11.2 推理配置

推理时设置：
- `jitter_noise = 0`（关闭探索噪声）
- `type_diversity_factor = 0`（不计算 diversity loss）
- `seq_expert_mode = "full_seq"`（保持 S1）

---

## 12. 计算量估算

假设 H=4096, L=512, B=8, 8 experts (3 flat + 3 seq + 2 MLP baseline), top_k=2：

| 组件 | 单层 FLOPs（近似） |
|------|-------------------|
| 原始 Time-MoE (8 MLP experts, top-2) | ~2 × (H × 2H × 2) × B×L ≈ 137G |
| Type-MoE flat 部分 (top-2 可能全为 flat) | ≤ 137G（同上） |
| Type-MoE seq 部分 (S1: 3 个 seq 专家全量计算) | +50~200G（取决于算子复杂度） |
| 共享专家 | ~H × 4H × 2 × B×L ≈ 137G（不变） |
| **预计总增量** | **+30% ~ +80%** |

> S1 的开销上界可控：即使所有 seq 专家都执行，也只相当于多跑 3 个轻量级算子，不是 3 个完整 Transformer。

---

## 13. 文件级实施清单

### 13.1 配置与入口

| 操作 | 文件 | 改动说明 |
|------|------|---------|
| 修改 | `time_moe/models/configuration_time_moe.py` | 新增 §8.1 所有字段 |
| 修改 | `main.py` | 新增 CLI 参数 / yaml 配置路径支持 |
| 修改 | `time_moe/runner.py` | 从配置创建 typed experts 与路由器 |

### 13.2 路由核心

| 操作 | 文件 | 改动说明 |
|------|------|---------|
| 修改 | `time_moe/models/modeling_time_moe.py` | `TimeMoeSparseExpertsLayer` → 支持 typed routing + S1 dispatch |
| 修改 | `time_moe/models/modeling_time_moe.py` | `load_balancing_loss_func` → 适配 typed routing aux |
| 新增 | `time_moe/models/typed_router_utils.py` | `typed_preselect`、`type_diversity_loss` 等纯函数 |

### 13.3 专家注册与实现

| 操作 | 文件 | 改动说明 |
|------|------|---------|
| 新增 | `time_moe/models/experts/__init__.py` | 包初始化 |
| 新增 | `time_moe/models/experts/base.py` | `BaseTokenExpert` 基类 |
| 新增 | `time_moe/models/experts/registry.py` | 注册表 + `build_expert_list` |
| 新增 | `time_moe/models/experts/mlp_temporal_expert.py` | 原 MLP 专家的 BaseTokenExpert 封装 |
| 新增 | `time_moe/models/experts/nbeats_token_expert.py` | N-BEATS flat 专家 |
| 新增 | `time_moe/models/experts/autoformer_token_expert.py` | Autoformer trend + cycle 专家 |
| 新增 | `time_moe/models/experts/fedformer_token_expert.py` | FEDformer cycle 专家 |
| 新增 | `time_moe/models/experts/anomaly_token_expert.py` | AnomalyTransformer anomaly 专家 |

### 13.4 日志与评估

| 操作 | 文件 | 改动说明 |
|------|------|---------|
| 修改 | `run_eval.py` | 增加 token 路由统计导出开关 |
| 新增 | `scripts/analyze_typed_routing.py` | 汇总 type 占比、专家利用率、aux 对齐检查 |

### 13.5 配置文件

| 操作 | 文件 | 改动说明 |
|------|------|---------|
| 新增 | `configs/typed_experts/base.yaml` | 默认 8 专家配置模板 |

---

## 14. 分阶段落地计划

| 阶段 | 内容 | 预计工期 | 关键交付物 | 验收标准 |
|------|------|---------|-----------|---------|
| **P1** | Typed Router 骨架 | 3-5 天 | `typed_preselect` + typed aux loss + diversity loss | typed_preselect 单元测试通过；aux 与 topk_indices 一致 |
| **P1.5** | 端到端烟测 | 1-2 天 | typed router + 原 MLP 专家 | 1000 步训练 loss 正常收敛（不劣于 baseline） |
| **P2** | 专家注册表 | 2-3 天 | registry + yaml 配置驱动 | 可按配置实例化 8 专家；state_dict 加载兼容 |
| **P3a** | NBeats (flat) | 2-3 天 | `NBeatsTokenExpert` | 单专家烟测 + 5000 步混合训练 |
| **P3b** | Autoformer (seq) | 3-5 天 | `AutoformerTrendExpert` + `AutoformerCycleExpert` | S1 dispatch 验证 + 混合训练 |
| **P3c** | FEDformer (seq) | 2-3 天 | `FedFormerCycleExpert` | FFT 精度验证（bf16 vs fp32） |
| **P3d** | AnomalyTransformer (seq) | 3-4 天 | `AnomalyTokenExpert` | anomaly attention 正确性验证 |
| **P4** | 联合训练 + 评测 | 5-7 天 | 全专家联合训练 + 基准对比 | Monash/ETT/Weather MSE ≥ 5% 改善 |

**总预计工期：22-32 天**

---

## 15. 验收标准

### 15.1 功能正确性

- [ ] 每 token 每类型最多 1 个候选进入 top-k
- [ ] `topk_indices` 与 `actual_k` 一致
- [ ] aux loss 的 `P_i` 使用预选前概率
- [ ] seq 专家在完整 `[B, L, H]` 上执行（S1）

### 15.2 精度目标

- [ ] Monash / ETT / Weather 基准上，相对 Time-MoE 基线 MSE 降低 ≥ 5%

### 15.3 效率约束

- [ ] 训练单步延迟 ≤ 原始 Time-MoE 的 2.0×
- [ ] GPU 显存 ≤ 原始 Time-MoE 的 1.5×

### 15.4 路由质量

- [ ] 专家利用率方差 < 阈值（typed routing 分散负载）
- [ ] `top_k=2` 下 type 占比不再结构性固定为 1/3
- [ ] 训练、推理、评估使用同一路由元信息来源

---

## 16. 风险与规避

| # | 风险 | 严重度 | 规避措施 |
|---|------|--------|---------|
| 1 | S1 下 seq 专家计算量不随选中 token 减少 | 中 | 限制每层 seq 专家数量（≤3）；`expert_mask.any()` 跳过未命中专家 |
| 2 | 类型约束导致组内马太效应 | 中 | jitter noise + aux loss 用预选前概率 + 每类型 ≥2 专家 |
| 3 | 异构专家梯度尺度不平衡 | 中 | 专家输出 RMSNorm + 可选独立 lr multiplier |
| 4 | KV-cache 推理下 seq 专家无法工作 | **高** | Seq Expert Hidden Cache 或推理时 flat 降级 |
| 5 | FFT 在 bf16 下数值不稳定 | 中 | 专家内部 upcast float32 |
| 6 | 预训练权重不兼容 | 低 | 保持 8 专家总数；MLP 专家复用原权重；新专家 zero-init |
| 7 | 类型标签先验与数据分布不匹配 | 中 | P4 分析 routing 统计，必要时调整类型划分 |
| 8 | Gradient checkpointing 与 seq 专家兼容性 | 中 | 验证 checkpoint 段边界行为 |
| 9 | 外部算子迁移后数值不稳定 | 中 | 分阶段接入，每新增一种先做单专家烟测 |

---

## 17. 回滚策略

| 回滚点 | 操作 | 回退至 |
|--------|------|--------|
| A | 仅关闭新型专家，保留 typed router | MLP-only + typed routing |
| B | 回退 typed router | 原生 Time-MoE top-k（`router_mode = "standard"`） |
| C | 关闭 custom_expert_specs | 恢复内置 `TimeMoeTemporalBlock` |

每个回滚点可通过配置切换，无需改代码。

---

## 附录 A：默认专家布局示意图

```
Layer N — TypedSparseExpertsLayer
├── Gate: Linear(H, 8)
├── Experts (8):
│   ├── [0] MLP-Trend-0      (flat, trend)   ← 预训练权重
│   ├── [1] NBeats-Trend      (flat, trend)   ← zero-init
│   ├── [2] Autoformer-Trend  (seq,  trend)   ← zero-init
│   ├── [3] MLP-Cycle-0       (flat, cycle)   ← 预训练权重
│   ├── [4] Autoformer-Cycle  (seq,  cycle)   ← zero-init
│   ├── [5] FedFormer-Cycle   (seq,  cycle)   ← zero-init
│   ├── [6] MLP-Anomaly-0     (flat, anomaly) ← 预训练权重
│   └── [7] Anomaly-Attn      (seq,  anomaly) ← zero-init
├── Shared Expert: TimeMoeTemporalBlock (常驻) ← 预训练权重
└── Shared Gate: Linear(H, 1)

Routing: softmax → (+jitter) → typed_preselect → top-2 → S1 dispatch → index_add_ → +shared
```

## 附录 B：关键术语表

| 术语 | 含义 |
|------|------|
| flat expert | 接受 `[M, H]` 输入的 pointwise 专家，与标准 MoE 稀疏 dispatch 兼容 |
| seq expert | 接受 `[B, L, H]` 输入的序列算子专家，需要完整序列上下文 |
| S1 | 全序列执行 + 选择性聚合：seq 专家始终处理完整序列，仅选中 token 位置的输出参与聚合 |
| typed_preselect | 类型内预选：每个类型仅保留组内概率最高的 1 个专家 |
| zero-init residual | 新专家输出投影初始化为 0，训练初期输出 ≈ 0，不干扰已收敛的模型 |
| jitter noise | 路由 logits 添加的随机噪声，防止 gate 过早固化到特定专家 |
