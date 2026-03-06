# 分阶段冻结训练策略（Phased Freeze Training）

> 更新时间：2026-03-06  
> 适用代码仓库：`C:\Users\kamil.liu\source\Type-MoE`  
> 对应实现文件：`time_moe/trainer/hf_trainer.py`、`time_moe/runner.py`  
> 配置入口：`configs/typed_experts/minimal_test.yaml` 中 `freeze_strategy: phased`

---

## 5.1 问题背景：微调异构 MoE 的两难困境

Type-MoE 的训练起点是已在大规模数据上预训练完成的 Time-MoE-50M。将其改造为异构 MoE 时，若直接对全部参数进行端到端微调，将面临两个相互耦合的优化困难：

**困难一：灾难性遗忘（Catastrophic Forgetting）**

5 个新引入的异构专家（E1/E2/E4/E5/E7）在初始阶段尚未经过任何训练，其输出是随机噪声（即便采用零初始化，梯度流入后早期更新仍不稳定）。这些随机输出通过路由权重混入隐状态，会产生巨大的梯度信号回传到预训练主干（Self-Attention、Embedding 等），导致原本已高质量收敛的预训练权重被快速覆写，性能出现断崖式退化。

**困难二：路由与专家的初始耦合死锁**

Gate 网络和专家网络在优化上高度相互依赖：Gate 需要根据专家的"真实能力"决定分发对象；专家需要接收到符合其归纳偏置的 token 分布后才能逐步习得对应特征。在两者均处于随机初始状态时同步训练，极易陷入"垃圾进垃圾出"的恶性循环，收敛到低质量局部极小。Lepikhin et al.（2021, GShard）指出 MoE 训练中路由与专家的联合优化存在固有不稳定性，并强调初始阶段专家分配的均衡对收敛至关重要，支持本文先稳定路由后激活专家的顺序设计。

> Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y., Krikun, M., Shazeer, N., & Chen, Z. (2021). GShard: Scaling giant models with conditional computation and automatic sharding. In *Proceedings of the 9th International Conference on Learning Representations* (ICLR). https://arxiv.org/abs/2006.16668

---

## 5.2 三阶段解冻协议

为解开上述死锁，本文实现了 `PhasedFreezeCallback`，在训练过程中动态控制参数的可训练性，将训练过程划分为三个阶段：

$$\text{Phase-A} \xrightarrow{\text{step}=T_A} \text{Phase-B} \xrightarrow{\text{step}=T_B} \text{Phase-C}$$

| 阶段 | 步数范围 | 可训练参数 | 冻结参数 |
|------|---------|-----------|---------|
| **Phase-A**（Gate 适配） | $[0,\ T_A)$ | 仅 gate 线性层（`ffn_layer.gate.weight`） | 主干 + 所有专家 |
| **Phase-B**（新专家预热） | $[T_A,\ T_B)$ | gate + 新异构专家（E1/E2/E4/E5/E7） | 主干 + 原始 MLP 专家 |
| **Phase-C**（全参数联调） | $[T_B,\ +\infty)$ | 全部参数 | 无 |

推荐配置：$T_A = 1000$，$T_B = 5000$（最小验证实验用 $T_A=20$，$T_B=50$）。

该三阶段设计遵循 Howard & Ruder（2018, ULMFiT）提出的"渐进解冻（Gradual Unfreezing）"范式：从最顶层开始逐层解冻，每次只更新已解冻的层，避免底层预训练特征被破坏。本文将其推广到 MoE 架构的"路由层 → 新专家 → 主干"解冻顺序。

> Howard, J., & Ruder, S. (2018). Universal language model fine-tuning for text classification. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics* (ACL), Vol. 1, 328–339. https://doi.org/10.18653/v1/P18-1031

---

## 5.3 各阶段的设计目标

### Phase-A：在"净化环境"中学习新路由规则

```
可训练：gate 线性层（36,864 参数，约占总参数 0.04%）
冻结：  全部专家权重 + 主干
新专家输出：≡ 0（零初始化保证）
```

此阶段模型行为在数值上等价于原始 Time-MoE（基座能力完整保留）。Gate 网络在无干扰的"纯净"环境中，专注学习两件事：适配 `typed_topk` 的类型约束路由规则（原始 `standard` 路由无此约束）；以及在类型多样性损失 $\mathcal{L}_{\text{div}}$ 的引导下，学习产生跨类型分散的概率分布。

`_identify_new_expert_indices` 函数从 `custom_expert_specs` 中读取所有标注了 `zero_init_output: true` 的专家索引，`_gate_patterns = [".gate."]` 用于精确匹配 gate 参数名，确保只解冻 gate 而不影响专家权重：

```python
def _apply_phase_a(self, model):
    _freeze_all(model)                              # 全部冻结
    _set_requires_grad(model, [".gate."], True)     # 仅解冻 gate
```

### Phase-B：在稳定路由下激活新专家

```
可训练：gate + 新异构专家（E1/E2/E4/E5/E7）
冻结：  主干（Attention、Embedding、LM Head）+ 原始 MLP 专家（E0/E3/E6）
```

此时 Gate 已具备区分类型的能力，能将趋势类 token 优先路由至趋势专家，周期类至周期专家，异常类至异常专家。在相对稳定的 token 分发下，零初始化的新专家开始接收梯度，从零逐步学习对应归纳偏置（移动平均趋势、FFT 周期、高斯注意力异常）。主干被冻结，避免了新专家早期不稳定输出对预训练表示的破坏；原始 MLP 专家（E0/E3/E6）也被冻结，为新专家提供稳定的对照基准。

```python
def _apply_phase_b(self, model):
    _freeze_all(model)
    _set_requires_grad(model,
        self._gate_patterns + self._new_expert_patterns, True)
    # self._new_expert_patterns = [".experts.1.", ".experts.2.", ".experts.4.", ...]
```

### Phase-C：全局协同精调

```
可训练：全部参数（97.869M）
```

Gate 和新专家均已具备一定能力后，解冻全部参数进行端到端联合优化。底层 Attention 的序列表示可以向上适配异构专家的输入期望，顶层预测头可以向下适配更丰富的隐状态。

```python
def _apply_phase_c(self, model):
    _unfreeze_all(model)   # 全部解冻，无约束训练
```

---

## 5.4 阶段切换机制

`PhasedFreezeCallback` 继承自 `transformers.TrainerCallback`，在每步训练开始时（`on_step_begin`）检查当前 `global_step`，按需触发阶段切换：

```python
def on_step_begin(self, args, state, control, model=None, **kwargs):
    step = state.global_step
    if   step < self.phase_a_end:  self._apply_phase_a(model)
    elif step < self.phase_b_end:  self._apply_phase_b(model)
    else:                          self._apply_phase_c(model)
```

每个 `_apply_phase_*` 方法内部有幂等保护（`if self._current_phase == "X": return`），避免重复操作带来的性能开销。此外，`runner.py` 中依据 `config.freeze_strategy` 字段决定注入哪种回调，支持 `phased`、`gate_only`、`none` 三种模式，向后完全兼容原始 Time-MoE 训练流程。

---

## 5.5 与零初始化残差的协同效应

分阶段冻结与零初始化残差在两个不同层面共同保护预训练知识，形成**双重防护**：

```
零初始化残差（数值层面）
  └─ 新专家输出 ≡ 0，隐状态数值上不受干扰

分阶段冻结（梯度层面）
  └─ Phase-A：主干梯度被阻断，反向传播不更新预训练权重
  └─ Phase-B：主干仍冻结，新专家在有限范围内安全学习
```

两者的保护范围互补：即使某次实验取消零初始化（如复用预训练 MLP 的 E0/E3/E6），分阶段冻结仍然保护主干；即使训练步数极短导致阶段切换不够平稳，零初始化也保证第一步的数值稳定性。

这与 LoRA 的设计哲学一致：新增模块在初始时刻对模型输出贡献为零，通过渐进训练实现向原始能力的平滑叠加。Hu et al.（2022）证明了新增模块零初始化后，微调初始时刻模型行为与原始预训练模型严格等价，梯度信号不会破坏已收敛的参数，是本文零初始化策略的直接理论根基。

> Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. In *Proceedings of the 10th International Conference on Learning Representations* (ICLR). https://arxiv.org/abs/2106.09685

---

## 5.6 归纳偏置的多层次施加

与传统通过数据增强或正则化施加归纳偏置的方式不同，Type-MoE 在三个层次上**同时**施加领域先验：

| 层次 | 机制 | 对应先验 | 激活时机 |
|------|------|---------|---------|
| **结构层**（专家算子拓扑） | 移动平均、FFT、高斯注意力的网络拓扑 | $Y_t = T_t + S_t + R_t$ 三分量独立存在 | $t=0$ 即存在，不依赖训练 |
| **路由层**（Gate 约束） | `typed_preselect` 强制跨类型选择 | 每个 token 的表示包含多种时序成分 | Phase-A 开始，Gate 学习期间逐步强化 |
| **初始化层**（零初始化） | 新专家训练起点为零贡献 | 预训练知识优先，新知识渐进融入 | $t=0$ 保护，Phase-B 后渐进激活 |

三层归纳偏置依次激活，形成从"保护预训练知识"到"渐进引入领域先验"的完整过渡路径。结构层的归纳偏置内嵌于算子拓扑，在参数初始化前即已存在，不依赖任何训练数据；路由层约束在 Phase-A 开始就通过 Gate 的梯度更新逐步体现；结构层的能力则在 Phase-B 之后随专家权重的实质性更新才真正发挥作用。

---

## 5.7 新旧训练策略对比

| 维度 | 原始 Time-MoE | Type-MoE（本文） |
|------|--------------|----------------|
| **训练模式** | 端到端全参数训练 | 三阶段渐进解冻 |
| **参数初始化** | 全部随机或预训练初始化 | 新专家输出投影零初始化 |
| **主干保护** | 无（直接微调） | Phase-A/B 期间主干冻结 |
| **路由适配顺序** | 路由与专家同步更新 | 先适配路由（Phase-A），再激活专家（Phase-B） |
| **可控粒度** | 无 | `phase_a_end`、`phase_b_end` 可独立配置 |
| **兼容性** | — | `freeze_strategy: none` 时完全等价于原始流程 |
