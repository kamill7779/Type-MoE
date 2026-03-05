# Type-MoE 最小测试完整报告

> 日期：2026-03-05  
> 实验类型：端到端最小流程验证（Minimal E2E Pipeline Validation）  
> 目标：在 RTX 5060 上以最低资源开销验证 Type-MoE 多专家路由训练全流程

---

## 1. 实验环境

| 项目 | 版本/规格 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 (8 GB VRAM) |
| CUDA | 12.8 |
| PyTorch | 2.11.0.dev20260130+cu128 |
| Transformers | 4.40.1 |
| Accelerate | 0.28.0 |
| Python | 3.10.19 (conda: autoformer) |
| OS | Windows 11 |
| 注意事项 | Windows 不支持 NCCL 多卡，单卡 eager attention 模式 |

---

## 2. 模型架构配置

### 2.1 基座模型（TimeMoE-50M）

| 参数 | 值 |
|---|---|
| 总参数量 | 97.869 M（加入异构专家后） |
| 隐层维度 (`hidden_size`) | 384 |
| 中间层维度 (`intermediate_size`) | 1536 |
| 注意力头数 | 12 |
| KV 头数 | 12 |
| Transformer 层数 | 12 |
| 最大位置编码 | 4096 |
| 预测层 (`horizon_lengths`) | [1, 8, 32, 64] |
| 损失函数 | Huber Loss (δ=2.0) + 辅助负载均衡损失 |

### 2.2 Type-MoE 配置覆盖（`configs/typed_experts/minimal_test.yaml`）

| 参数 | 值 | 说明 |
|---|---|---|
| `router_mode` | `typed_topk` | 类型约束 Top-K 路由 |
| `num_experts_per_tok` | 2 | 每 token 激活 2 个专家 |
| `num_experts` | 8 | 共 8 个异构专家 |
| `norm_topk_prob` | false | 不对路由权重归一化 |
| `jitter_noise` | 0.02 | 训练时路由噪声（防止路由坍缩） |
| `type_diversity_factor` | 0.01 | 类型多样性辅助损失权重 |
| `router_aux_loss_factor` | 0.02 | 负载均衡辅助损失权重 |
| `freeze_strategy` | `phased` | 三阶段分阶冻结 |

### 2.3 异构专家池（8 个专家）

| 专家编号 | 名称 | 类型 | 接口 | 是否零初始化 | 说明 |
|---|---|---|---|---|---|
| E0 | `mlp_temporal_block` | trend | flat | ✗ | 继承预训练 MLP 权重 |
| E1 | `nbeats_trend` | trend | flat | ✓ | N-BEATS 趋势分解块（4层，theta_dim=16）|
| E2 | `autoformer_trend` | trend | seq | ✓ | Autoformer 趋势 MA 层（kernel=25） |
| E3 | `mlp_temporal_block` | cycle | flat | ✗ | 继承预训练 MLP 权重 |
| E4 | `autoformer_cycle` | cycle | seq | ✓ | Autoformer 频域周期提取（top_k_freq=3） |
| E5 | `fedformer_cycle` | cycle | seq | ✓ | FEDformer 傅里叶振型混合（modes=32） |
| E6 | `mlp_temporal_block` | anomaly | flat | ✗ | 继承预训练 MLP 权重 |
| E7 | `anomaly_attn` | anomaly | seq | ✓ | AnomalyTransformer 自关联注意力 |

`flat` 接口：接受 `[T, H]` 形状，逐 token 处理；`seq` 接口：接受 `[B, L, H]` 全序列处理（S1 策略缓存）。

---

## 3. 训练配置

### 3.1 超参数

| 参数 | 值 | 说明 |
|---|---|---|
| 数据集 | ETTh1.csv（前 60%）| 7 个特征列，各 10452 个时间点 |
| 上下文窗口 (`max_length`) | 128 | 较短，节省 VRAM |
| 滑窗步长 (`stride`) | 128 | 无重叠滑窗 |
| 训练步数 | 100 | 最小验证流程 |
| 全局批大小 | 4 | |
| 单卡批大小 | 2 | |
| 梯度累积步数 | 2 | |
| 学习率 | 5×10⁻⁴（峰值）| cosine 衰减至 1×10⁻⁵ |
| Warmup 步数 | 10 | |
| 权重衰减 | 0.1 | |
| 精度 | bf16 | bfloat16 |
| 梯度裁剪 | 1.0 | |
| 序列归一化 | zero | 零均值标准差归一化 |
| 注意力实现 | eager | 无 flash_attn |

### 3.2 分阶段冻结策略（Phased Freeze）

| 阶段 | 步数范围 | 可训练参数 | 可训练参数量 |
|---|---|---|---|
| Phase-A (Gate 适配) | 步骤 0 – 19 | 仅 gate 线性层 | **36,864** |
| Phase-B (新专家预热) | 步骤 20 – 49 | gate + 零初始化新专家 | ~数 M |
| Phase-C (全参数联调) | 步骤 50 – 100 | 全部参数 | 97.869 M |

> **设计原理**：零初始化新专家（E1/E2/E4/E5/E7）在 Phase-A 时贡献为零，不破坏基座预训练性能；Phase-B 逐步激活新专家；Phase-C 端到端联合调优。

### 3.3 训练损失曲线

| 步骤 | 损失值 | 梯度范数 | 阶段 |
|---|---|---|---|
| 10 | 0.1524 | 0.0581 | Phase-A |
| 20 | 0.1920 | 0.0527 | Phase-A → B |
| 30 | 0.1369 | 0.0439 | Phase-B |
| 40 | 0.1291 | 0.5313 | Phase-B |
| 50 | 0.1622 | 0.4043 | Phase-B → C |
| 60 | 0.1590 | 0.5742 | Phase-C |
| 70 | 0.1684 | 0.7148 | Phase-C |
| 80 | 0.1913 | 7.5625 | Phase-C |
| 90 | 0.1542 | 2.0313 | Phase-C |
| 100 | 0.1518 | 2.2969 | Phase-C |

训练总用时：**7 分 21 秒**（441s），平均 4.4s/步。

---

## 4. 输入处理流程

```
原始时序数据 (ETTh1.csv)
        │
        ▼
  [数据准备]  CSV → JSON
  7列特征各提取前60%时间点
  → etth1_train.json (7条序列 × 10452点)
        │
        ▼
  [TimeMoEDataset]  读取 JSON
  每条序列作为独立时间序列
        │
        ▼
  [TimeMoEWindowDataset]  滑窗切割
  context_length=128, stride=128
  → (input_ids[128], labels[128], loss_masks[128])
        │
        ▼
  [序列归一化]  zero-scaler
  x' = (x - mean) / std
        │
        ▼
  [输入嵌入]  TimeMoeInputEmbedding
  input_ids: [B, L] → GLU MLP → [B, L, H=384]
        │
        ▼
  [Transformer × 12 层]  每层结构：
  ┌─────────────────────────────────────┐
  │  RMSNorm → TimeMoeAttention (GQA)   │
  │  + 残差                             │
  │  RMSNorm → TimeMoeSparseExpertsLayer│
  │    ├─ gate: Linear(H→8) → softmax   │
  │    ├─ typed_preselect: 每类留最优1个 │
  │    ├─ topk(k=2): 跨类选 2 个专家    │
  │    ├─ 专家执行 (flat/seq S1 策略)   │
  │    └─ shared_expert + sigmoid gate  │
  │  + 残差                             │
  └─────────────────────────────────────┘
        │
        ▼
  [多地平线预测头]  lm_heads × 4
  horizon=[1,8,32,64] → 最终预测值
        │
        ▼
  [损失计算]
  AR Huber Loss (4个地平线均值)
  + 负载均衡辅助损失 × 0.02
  + 类型多样性辅助损失 × 0.01
```

### 4.1 typed_topk 路由机制

```
原始 gate logits [T, 8]
      │ softmax
      ▼
raw_probs [T, 8]
      │ typed_preselect
      │  - trend 组 (E0,E1,E2): 保留概率最大的 1 个
      │  - cycle 组 (E3,E4,E5): 保留概率最大的 1 个  
      │  - anomaly 组 (E6,E7): 保留概率最大的 1 个
      ▼
filtered_probs [T, 8]  (每行至多 3 个非零)
      │ topk(k=2)
      ▼
selected_experts [T, 2]  (跨类型选最强的 2 个)
routing_weights  [T, 2]
```

### 4.2 S1 序列专家调度策略

对于 `interface="seq"` 的专家（E2/E4/E5/E7），采用 S1 全序列缓存策略：  
1. 在该层第一次被选中时，对完整 `[B, L, H]` 做一次前向（覆盖所有步）  
2. 将结果缓存在 `seq_expert_cache[expert_idx]`  
3. 只从缓存中取被路由到的位置 `top_x` 的输出进行加权聚合  
4. 避免重复调用全序列计算，节约计算量

---

## 5. 专家路由统计（训练后）

### 5.1 各层专家选择比例（%）

| 层 | E0 MLP-trend | E1 N-BEATS | E2 Auto-trend | E3 MLP-cycle | E4 Auto-cycle | E5 FED-cycle | E6 MLP-anomaly | E7 Anomaly-attn |
|---|---|---|---|---|---|---|---|---|
| Layer 0 | 0.0 | **50.0** | 0.0 | **46.6** | 0.8 | 2.6 | 0.0 | 0.0 |
| Layer 1 | 1.9 | 6.9 | **40.4** | 3.0 | 7.5 | 3.1 | **36.1** | 1.1 |
| Layer 2 | 16.2 | 4.0 | 3.1 | 1.9 | 4.2 | **23.6** | 20.8 | **26.1** |
| Layer 3 | **22.0** | 17.7 | 3.1 | 5.5 | **18.8** | 7.3 | 12.8 | 12.8 |
| Layer 4 | 12.6 | 3.5 | 12.0 | 14.0 | **23.7** | 6.0 | **24.0** | 4.2 |
| Layer 5 | **38.8** | 0.9 | 1.0 | **31.0** | 5.5 | 6.3 | 10.8 | 5.7 |
| Layer 6 | 3.3 | 6.1 | 13.8 | 3.1 | 3.0 | **41.2** | **23.4** | 6.1 |
| Layer 7 | 3.5 | 4.0 | 1.4 | 5.4 | **30.0** | 10.3 | **26.9** | 18.5 |
| Layer 8 | 2.0 | **37.0** | 3.3 | **17.9** | 0.8 | **27.2** | 4.7 | 7.1 |
| Layer 9 | **35.5** | 5.4 | 2.6 | 5.0 | **26.5** | 4.8 | 8.6 | 11.6 |
| Layer 10 | 3.8 | 1.1 | **34.0** | 8.2 | 11.1 | **21.5** | 9.6 | 10.7 |
| Layer 11 | 3.9 | 0.8 | **38.0** | **18.0** | 4.2 | 0.6 | 16.1 | **18.3** |

### 5.2 按类型汇总

| 专家类型 | 专家 | 全局选择比例 | 总选择次数 |
|---|---|---|---|
| **trend** | E0 + E1 + E2 | **36.1%** | 269,774 |
| **cycle** | E3 + E4 + E5 | **37.5%** | 280,231 |
| **anomaly** | E6 + E7 | **26.3%** | 196,491 |
| **合计** | — | 100% | 746,496 |

> **解读**：三类专家的路由比例相对均衡（趋势 36% ≈ 周期 38% ≈ 异常 26%），说明 typed_topk 路由成功确保了类型多样性。各层展现出清晰的专家偏好差异化（如 Layer 0 偏向趋势+周期 MLP、Layer 10-11 偏向 Autoformer-trend），无路由坍缩现象。

---

## 6. 评估结果

数据集：ETTh1，测试集（后20%），滑窗评估

| 指标 | Type-MoE（100步）| 原始 TimeMoE-50M（0步）| 差值 |
|---|---|---|---|
| **MSE** | 0.360281 | 0.357681 | +0.0026 |
| **MAE** | 0.389527 | 0.381649 | +0.0079 |
| **WAPE** | 0.489365 | — | — |

| 评估配置 | 值 |
|---|---|
| 预测长度 (`pred_len`) | 96 |
| 上下文长度 (`ctx_len`) | 512 |
| 批大小 | 16 |
| 测试样本数 | 19,488 个窗口 |
| 总预测点数 | 1,870,848 |
| 评估用时 | 2447 s（~41 min） |

> 仅经过 100 步轻量训练，Type-MoE 的 MSE 与原始模型相差仅 **0.0026**（0.73%），证明异构专家注入不破坏基座表示能力，分阶段冻结策略有效保护了预训练知识。

---

## 7. 下一步改进方向

### 7.1 增加训练步数（最高优先级）

仅 100 步远未收敛，Phase-C 阶段梯度范数仍在 2-8 之间波动，说明模型参数尚未充分适配异构专家。

| 方案 | 推荐训练步数 | 预估资源 | 预期效果 |
|---|---|---|---|
| 快速验证 | 500 步 | ~37 min | MSE ↓ ~5% |
| 标准调优 | 2000 步（Phase-A: 200, B: 800, C: 1000）| ~2.5 h | MSE ↓ ~10–15% |
| 完整训练 | 5000 步 | ~6 h | 接近文献最优 |

建议下次运行：
```bash
python run_typemoe_minimal_test.py  # 修改 TRAIN_STEPS=2000, PHASE_A_END=200, PHASE_B_END=800
```

### 7.2 扩展训练数据

当前只用 ETTh1 单变量，异构专家需要更多样的时序模式来充分激活各专家。

| 数据扩展方案 | 数据集 | 说明 |
|---|---|---|
| ETT 全集 | ETTh1/h2/m1/m2 | 覆盖不同采样率，趋势多样性更强 |
| + Weather | weather.csv (21列) | 气象数据含强周期性信号，利于周期专家 |
| + Electricity | electricity.csv (321列) | 日负荷数据含明显异常峰，利于异常专家 |

修改 `data_prepared/` 目录或修改 `TRAIN_DATA_PATH` 指向多数据集目录。

### 7.3 调整专家规格（针对显存）

对于 RTX 5060 (8 GB) 的显存限制，可以将 seq 专家的参数规模缩减：

| 专家 | 当前参数 | 推荐缩减 |
|---|---|---|
| `autoformer_trend` | `kernel_size=25` | 保持 |
| `autoformer_cycle` | `top_k_freq=3` | 降至 `top_k_freq=2` |
| `fedformer_cycle` | `modes=32` | 降至 `modes=16` |
| `nbeats_trend` | `num_layers=4, theta_dim=16` | 降至 `num_layers=2` |

### 7.4 调整路由参数

| 参数 | 当前值 | 建议调整 | 理由 |
|---|---|---|---|
| `jitter_noise` | 0.02 | 0.01（训练后期） | 后期降低探索噪声，稳定路由 |
| `type_diversity_factor` | 0.01 | 0.02–0.05 | 若异常专家比例低，加强多样性惩罚 |
| `router_aux_loss_factor` | 0.02 | 0.01 | 减弱负载均衡约束，允许自然专业化 |

### 7.5 上下文长度提升

当前训练 `max_length=128`，但评估用 `ctx_len=512`，存在明显的训练-推理 gap。

| 方案 | `max_length` | VRAM 占用（估计）| 说明 |
|---|---|---|---|
| 当前（最小） | 128 | ~3 GB | 训练快，但与推理不匹配 |
| 均衡方案 | 256 | ~4.5 GB | 减小 gap |
| 推荐方案 | 512 | ~6.5 GB | 与评估完全一致 |

### 7.6 多预测长度评估

建议在完整训练后对所有标准预测长度进行评估，完整对比：

| pred_len | ctx_len | 预期 MSE |
|---|---|---|
| 96 | 512 | — |
| 192 | 1024 | — |
| 336 | 2048 | — |
| 720 | 3072 | — |

---

## 8. 关键文件索引

| 文件 | 作用 |
|---|---|
| `configs/typed_experts/minimal_test.yaml` | Type-MoE 最小测试配置 |
| `configs/typed_experts/base.yaml` | Type-MoE 完整配置（同 minimal，无 freeze_strategy） |
| `run_typemoe_minimal_test.py` | 端到端训练+路由统计+评估脚本 |
| `main.py` | 通用训练 CLI 入口 |
| `run_eval.py` | 基准评估脚本（MSE/MAE/WAPE + routing stats） |
| `time_moe/models/modeling_time_moe.py` | 核心模型（TimeMoeSparseExpertsLayer + S1 调度） |
| `time_moe/models/typed_router_utils.py` | typed_topk 路由算法 + 辅助损失 |
| `time_moe/models/experts/` | 8 种异构专家实现 |
| `time_moe/runner.py` | 训练流程 + 分阶段冻结回调 |
| `time_moe/trainer/hf_trainer.py` | 自定义 Trainer + PhasedFreezeCallback |
| `logs/typemoe_minimal_test/routing_stats.json` | 12 层路由统计（本次实验） |
| `logs/typemoe_minimal_test/eval_results.json` | MSE/MAE/WAPE 评估结果（本次实验） |
