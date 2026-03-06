# 02 — Seq 专家推理正确性修复方案

> 日期：2026-03-07
> 目标：以**正确性最大化**为前提，修复 seq 专家在自回归推理中的上下文丢失 + flat_fallback 未注册两个缺陷

---

## 1. 问题定义

### 1.1 Seq 专家在 AR 步骤中丢失上下文

`model.generate()` 使用 KV-cache + multi-horizon 自回归生成。Attention 通过 KV-cache 保留
完整上下文，但 seq 专家（Autoformer / FEDformer / AnomalyAttn）只看到当前 chunk 的
`hidden_states`，无法获得历史信息。

以 `pred_len=96, ctx_len=512` 为例：

```
Step 1 (prefill): model 输入 512 tokens → seq 专家 forward([B,512,H]) ✓
    → lm_head(h=64) 输出 64 tokens

Step 2 (AR):      model 输入 64 tokens  → seq 专家 forward([B, 64,H]) ✗
    → AutoformerTrend: MA(kernel=25) 在 64 点上做，边界 padding 影响 ~40%
    → AutoformerCycle: FFT 只有 33 频率 bin（vs prefill 257 bin，分辨率 ↓87%）
    → FEDformer(modes=32): 32/33 modes ≈ 全通滤波，失去选择性
    → AnomalyAttn: 64×64 注意力（vs 512×512，缩小 64 倍）
```

`pred_len=720` 更严重，最后几个 AR 步只有 **8 tokens**：FFT 仅 5 个频率 bin，MA 以
padding 为主。

### 1.2 `forward_flat_fallback` 懒初始化未注册

```python
# base.py 当前实现
def forward_flat_fallback(self, x):
    if self._flat_fallback is None:
        self._flat_fallback = nn.Linear(H, H, bias=False)  # 运行时动态创建
        nn.init.zeros_(self._flat_fallback.weight)
    return self._flat_fallback(x)
```

- `_flat_fallback` 不在 `nn.Module` 注册的子模块中 → 不在 `model.parameters()` 中
- 不参与训练、不会被 `model.save_pretrained()` 保存
- 每次加载模型后重新创建（零权重），永远输出 0

---

## 2. 修复方案

### 2.1 修复 A：SeqContextBuffer — 让 seq 专家始终看到完整上下文

**核心思想**：在每层 `TimeMoeSparseExpertsLayer` 中维护一个 hidden_states 缓冲区。
推理时 seq 专家在缓冲区上做 forward，然后只取尾部 `n_new` 个 token 的输出。

#### 数据流

```
                                    ┌───────────────────────────┐
prefill (Step 1):                   │  _seq_ctx_buffer          │
  hidden_states [B,512,H] ────────▶│  容量 = seq_expert_ctx_len│
  seq_expert(buffer) → [B,512,H]   │  当前: [B,512,H]          │
  取全部 512 tokens 输出            └───────────────────────────┘
                                                │
AR Step 2:                                      ▼ 追加 + 截断
  hidden_states [B,64,H] ──────▶ buffer = cat([old 448], [new 64]) = [B,512,H]
  seq_expert(buffer) → [B,512,H]
  只取最后 64 tokens 输出            ← top_x 索引不变，S1 cache 只存尾部

AR Step 12 (8 tokens):
  hidden_states [B,8,H] ───────▶ buffer = cat([old 504], [new 8]) = [B,512,H]
  seq_expert(buffer) → [B,512,H]
  只取最后 8 tokens 输出             ← FFT 仍有 257 频率 bin ✓
```

#### 关键实现细节

1. **缓冲区仅在推理时启用**（`not self.training`）
2. **路由仍基于当前 hidden_states** — 不改变路由决策
3. **S1 cache 只存储尾部** — seq 专家对 buffer 做完整 forward，但只截取最后 `sequence_length` 个 token 存入 `seq_expert_cache`，使 `top_x` 索引保持不变
4. **`_use_seq_fallback` 条件更新** — 当 buffer 长度 > 1 时不再回退到 flat_fallback
5. **生成前 reset** — 在 `_greedy_search()` 开始时清空各层缓冲区

#### 变更文件

| 文件 | 变更 |
|---|---|
| `time_moe/models/configuration_time_moe.py` | 新增 `seq_expert_context_len` 参数 |
| `time_moe/models/modeling_time_moe.py` | `TimeMoeSparseExpertsLayer` 添加缓冲区逻辑 |
| `time_moe/models/ts_generation_mixin.py` | `_greedy_search()` 前 reset 缓冲区 |
| `configs/typed_experts/*.yaml` | 添加 `seq_expert_context_len: 512` |

### 2.2 修复 B：FixFlatFallback — 正式注册 flat_fallback 模块

**核心思想**：将 `_flat_fallback` 从懒初始化改为在 `__init__` 中正式注册为 `nn.Module` 子模块。

#### 变更

1. `BaseTokenExpert.__init__` 接收 `hidden_size` 参数
2. seq 类型专家在 `__init__` 中创建 `self.flat_fallback = nn.Linear(H, H, bias=False)`
3. 零初始化权重，与 `zero_init_output` 策略一致
4. 所有子类 `super().__init__(hidden_size=hidden_size)` 传参

#### 变更文件

| 文件 | 变更 |
|---|---|
| `time_moe/models/experts/base.py` | 重写 `__init__` 和 `forward_flat_fallback` |
| `time_moe/models/experts/autoformer_token_expert.py` | `super().__init__` 传 hidden_size |
| `time_moe/models/experts/fedformer_token_expert.py` | 同上 |
| `time_moe/models/experts/anomaly_token_expert.py` | 同上 |
| `time_moe/models/experts/nbeats_token_expert.py` | 同上（flat 类型，不创建 fallback） |
| `time_moe/models/experts/mlp_temporal_block_expert.py` | 同上（flat 类型） |

---

## 3. 实施清单

```
Phase 1 — FixFlatFallback (~30 min):
  [ ] base.py: __init__ 接收 hidden_size，seq 类型注册 flat_fallback
  [ ] 4 个 seq 专家子类: super().__init__(hidden_size=...)
  [ ] 2 个 flat 专家子类: super().__init__() 不变或传参（不创建 fallback）

Phase 2 — SeqContextBuffer (~2 h):
  [ ] configuration_time_moe.py: 新增 seq_expert_context_len 参数
  [ ] modeling_time_moe.py: TimeMoeSparseExpertsLayer 添加缓冲区
  [ ] ts_generation_mixin.py: _greedy_search 前 reset
  [ ] configs/*.yaml: 添加 seq_expert_context_len

Phase 3 — 验证 (~30 min):
  [ ] 运行 run_typemoe_minimal_test.py 验证训练不受影响
  [ ] 运行评估验证推理正确性
  [ ] git commit
```
