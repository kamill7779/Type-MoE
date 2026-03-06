# 01 — Type-MoE 部署与测试指南

> 最后更新：2026-03-07  
> 适用范围：H20-96G 生产训练、RTX 5060 开发验证

---

## 目录

1. [环境准备](#1-环境准备)
2. [数据集准备](#2-数据集准备)
3. [快速验证（开发机）](#3-快速验证开发机)
4. [生产训练（H20-96G）](#4-生产训练h20-96g)
5. [仅评估已有模型](#5-仅评估已有模型)
6. [文件结构说明](#6-文件结构说明)
7. [配置参数参考](#7-配置参数参考)
8. [故障排除](#8-故障排除)

---

## 1. 环境准备

### 1.1 创建 Conda 环境

```bash
conda create -n typemoe python=3.10 -y
conda activate typemoe
```

### 1.2 安装依赖

```bash
pip install -r requirements.txt
```

核心依赖版本要求：

| 包名 | 版本 | 说明 |
|---|---|---|
| torch | ≥ 2.1 | 需 CUDA 支持 |
| transformers | 4.40.1 | 必须此版本，兼容自定义模型 |
| accelerate | ≥ 0.28 | HuggingFace Trainer 依赖 |
| safetensors | ≥ 0.4 | 模型权重加载 |
| datasets | ≥ 2.18 | 数据集工具 |
| pyyaml | ≥ 6.0 | 配置文件解析 |
| pandas | ≥ 1.5 | CSV 数据读取 |
| numpy | ≥ 1.24 | 数值计算 |

### 1.3 验证 GPU

```bash
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory // 1024**3, 'GB')"
```

### 1.4 （可选）Linux / H20 环境额外步骤

```bash
# 安装 flash-attn（H20 原生支持）
pip install flash-attn --no-build-isolation

# 安装 NCCL（多卡分布式训练）
# 通常随 PyTorch CUDA 版自带，无需额外安装
```

> **Windows 注意事项**：Windows 不支持 NCCL，仅能单卡运行。注意力实现只能使用 `eager` 模式。

---

## 2. 数据集准备

### 2.1 数据集位置

数据集应放置在项目根目录下 `data/` 文件夹：

```
Type-MoE/
  data/
    ETT-small/
      ETTh1.csv
      ETTh2.csv
      ETTm1.csv
      ETTm2.csv
    weather.csv
    electricity.csv
```

### 2.2 获取数据集

**方法 1：从 Autoformer 项目复制（如果已有）**

```bash
# Linux/Mac
cp -r /path/to/Autoformer/dataset/ETT-small data/ETT-small
cp /path/to/Autoformer/dataset/weather/weather.csv data/
cp /path/to/Autoformer/dataset/electricity/electricity.csv data/

# Windows (PowerShell)
Copy-Item -Recurse E:\Project\Autoformer\dataset\ETT-small data\ETT-small
Copy-Item E:\Project\Autoformer\dataset\weather\weather.csv data\
Copy-Item E:\Project\Autoformer\dataset\electricity\electricity.csv data\
```

**方法 2：从互联网下载**

ETT 数据集来自论文 *Informer* 的官方仓库：
- GitHub: https://github.com/zhouhaoyi/ETDataset

Weather/Electricity 数据集来自 *Autoformer* 仓库：
- GitHub: https://github.com/thuml/Autoformer

### 2.3 数据集概览

| 数据集 | 文件 | 列数 | 行数 | 频率 | 主要特征 |
|---|---|---|---|---|---|
| ETTh1 | ETTh1.csv | 7 | 17,420 | 小时 | 电力变压器温度 (趋势+日周期) |
| ETTh2 | ETTh2.csv | 7 | 17,420 | 小时 | 同上，不同变电站 |
| ETTm1 | ETTm1.csv | 7 | 69,680 | 15分钟 | 高频版 ETTh1 |
| ETTm2 | ETTm2.csv | 7 | 69,680 | 15分钟 | 高频版 ETTh2 |
| Weather | weather.csv | 21 | 52,696 | 10分钟 | 气象多变量 (强周期信号) |
| Electricity | electricity.csv | 321 | 26,304 | 小时 | 用电负荷 (含异常峰值) |

### 2.4 基座模型

确保 `TimeMoE-50M/` 文件夹位于项目根目录下，包含以下文件：

```
TimeMoE-50M/
  config.json
  configuration_time_moe.py
  modeling_time_moe.py
  ts_generation_mixin.py
  model.safetensors
  generation_config.json
```

---

## 3. 快速验证（开发机）

适用于 **8-16 GB VRAM** 的消费级 GPU，用于验证流程正确性。

### 3.1 最小端到端测试（100 步）

```bash
python run_typemoe_minimal_test.py
```

- **用时**：~7 分钟 (RTX 5060 8GB)
- **输出**：`logs/typemoe_minimal_test/`
  - `routing_stats.json` — 12 层路由统计
  - `eval_results.json` — ETTh1 pred_len=96 的 MSE/MAE/WAPE
- **预期 MSE**：~0.36（与原始 TimeMoE-50M 的 0.357 接近）

### 3.2 开发 profile 快速训练（500 步）

```bash
python run_typemoe_production.py --profile dev
```

- **用时**：~30 分钟
- **输出**：`logs/typemoe_dev/`
- **预期 MSE**：~0.34–0.35

### 3.3 验证要点

1. 训练日志应显示三个阶段切换：
   ```
   PhasedFreezeCallback: entered Phase-A (gate only)
   PhasedFreezeCallback: entered Phase-B (gate + new experts)
   PhasedFreezeCallback: entered Phase-C (all params)
   ```
2. `routing_stats.json` 中三类专家比例应大致均衡（各 25–40%）
3. 评估 MSE 应不高于原始 TimeMoE-50M 太多（±10% 以内）

---

## 4. 生产训练（H20-96G）

### 4.1 默认生产运行

```bash
python run_typemoe_production.py
```

等价于：

```bash
python run_typemoe_production.py --profile h20
```

**默认配置**：

| 参数 | 值 | 说明 |
|---|---|---|
| 训练步数 | 5,000 | 完整训练 |
| 序列长度 | 512 | 与评估对齐 |
| 滑窗步长 | 256 | 50% 重叠 |
| 微批大小 | 16 | 单卡批大小 |
| 全局批大小 | 64 | 梯度累积 4 步 |
| 学习率 | 2×10⁻⁴ | cosine → 1×10⁻⁵ |
| Phase-A | 0 → 500 步 | 仅 gate 可训 |
| Phase-B | 500 → 2000 步 | gate + 新专家 |
| Phase-C | 2000+ 步 | 全参数联调 |
| 精度 | bf16 | H20 原生支持 |
| 注意力 | flash_attention_2 | 自动检测 |
| 梯度检查点 | 启用 | 节省显存 |
| 数据集 | ETTh1/h2, ETTm1/m2, Weather, Electricity | 6 个数据集 |
| 预测长度 | 96, 192, 336, 720 | 4 种标准长度 |

### 4.2 预计时间

| 阶段 | 预计用时 | 说明 |
|---|---|---|
| 数据准备 | ~1 分钟 | CSV → JSON 一次性 |
| 训练 (5000 步) | ~3–4 小时 | H20-96G |
| 路由统计 | ~5 分钟 | 5 条序列 forward |
| 评估 (6×4) | ~6–8 小时 | Electricity 最耗时 (321 列) |
| **合计** | **~10–12 小时** | |

### 4.3 自定义参数

```bash
# 只训练 ETTh1+Weather，2000 步
python run_typemoe_production.py --datasets ETTh1 weather --train_steps 2000

# 更大批次 + 更短训练
python run_typemoe_production.py --micro_batch 32 --global_batch 128 --train_steps 3000

# 指定自定义输出目录
python run_typemoe_production.py --output_dir logs/experiment_v2
```

### 4.4 标准调优 Profile

适用于 24–48 GB VRAM (A100-40G, RTX 4090 等)：

```bash
python run_typemoe_production.py --profile standard
```

| 参数 | 值 |
|---|---|
| 训练步数 | 2,000 |
| 微批大小 | 8 |
| 数据集 | ETTh1, ETTh2, Weather |
| 预测长度 | 96, 192, 336 |

### 4.5 检查点说明

训练过程中会自动保存中间检查点：

- 保存频率：每 1/5 总步数（H20 默认每 1000 步）
- 保留数量：最近 3 个
- 位置：`{output_dir}/checkpoint-{step}/`
- 最终模型：`{output_dir}/` 根目录下的 `model.safetensors` + `config.json`

---

## 5. 仅评估已有模型

```bash
# 评估默认输出目录的模型
python run_typemoe_production.py --eval_only

# 指定模型目录
python run_typemoe_production.py --eval_only --model_dir logs/typemoe_h20

# 评估指定数据集和预测长度
python run_typemoe_production.py --eval_only --model_dir logs/typemoe_h20 \
    --datasets ETTh1 ETTh2 --pred_lens 96 192

# 使用 run_eval.py 单数据集评估 (支持路由统计导出)
python run_eval.py -m logs/typemoe_h20 -d data/ETT-small/ETTh1.csv -p 96 \
    --export_routing_stats --routing_stats_path logs/routing_eval.json
```

---

## 6. 文件结构说明

### 6.1 核心脚本

| 文件 | 功能 | 使用场景 |
|---|---|---|
| `run_typemoe_production.py` | **生产训练+评估流水线** | H20/A100 大规模训练 |
| `run_typemoe_minimal_test.py` | 最小端到端验证 | 开发机快速测试 |
| `run_eval.py` | 单数据集基准评估 | 独立评估 + 路由统计 |
| `main.py` | 通用 CLI 训练入口 | 自定义训练（无 Type-MoE） |

### 6.2 配置文件

| 文件 | 说明 |
|---|---|
| `configs/typed_experts/h20_production.yaml` | H20 生产配置（phased freeze） |
| `configs/typed_experts/minimal_test.yaml` | 最小测试配置 |
| `configs/typed_experts/base.yaml` | 基准配置（无 freeze） |

### 6.3 模型代码

| 路径 | 说明 |
|---|---|
| `time_moe/models/modeling_time_moe.py` | 核心模型 (TimeMoeSparseExpertsLayer + S1 调度) |
| `time_moe/models/typed_router_utils.py` | typed_topk 路由算法 + 辅助损失 |
| `time_moe/models/experts/` | 8 种异构专家实现 |
| `time_moe/runner.py` | 训练流程 + 配置覆盖 |
| `time_moe/trainer/hf_trainer.py` | Trainer + PhasedFreezeCallback |
| `time_moe/datasets/` | 数据集加载 + 滑窗处理 |

### 6.4 输出目录结构

```
logs/typemoe_h20/
  ├── run_config.json          # 完整运行配置
  ├── config.json              # 模型配置
  ├── model.safetensors        # 最终模型权重
  ├── routing_stats.json       # 12 层路由统计
  ├── eval_results.json        # 多数据集评估结果
  ├── tb_logs/                 # TensorBoard 日志
  ├── data_prepared/
  │   └── train_data.json      # 合并后的训练数据
  └── checkpoint-*/            # 中间检查点
```

---

## 7. 配置参数参考

### 7.1 Profile 对比

| 参数 | `dev` | `standard` | `h20` |
|---|---|---|---|
| 目标 GPU | 8-16 GB | 24-48 GB | 96 GB |
| 训练步数 | 500 | 2,000 | 5,000 |
| 序列长度 | 256 | 512 | 512 |
| 微批大小 | 2 | 8 | 16 |
| 全局批大小 | 8 | 32 | 64 |
| 学习率 | 5×10⁻⁴ | 3×10⁻⁴ | 2×10⁻⁴ |
| Phase-A 终止 | 50 | 200 | 500 |
| Phase-B 终止 | 200 | 800 | 2,000 |
| 注意力 | eager | eager | auto (flash) |
| 梯度检查点 | 否 | 否 | 是 |
| 数据集 | ETTh1 | ETTh1/h2 + Weather | 6 个全集 |
| 预测长度 | 96 | 96/192/336 | 96/192/336/720 |
| 预计用时 | ~30 min | ~3 hr | ~10-12 hr |

### 7.2 分阶段冻结策略

```
Phase-A (Gate 适配)     Phase-B (新专家预热)     Phase-C (全参数联调)
├─────────────────────┤├─────────────────────┤├───────────────────────┤
step 0                 phase_a_end            phase_b_end             train_steps

可训练参数:
  A: gate 线性层 (~37K)
  B: gate + 零初始化新专家 (~数百 K)
  C: 全部 (~98M)
```

### 7.3 typed_topk 路由参数

| 参数 | 默认值 | 说明 | 调优建议 |
|---|---|---|---|
| `jitter_noise` | 0.02 | 路由噪声，防止坍缩 | 训练后期可降至 0.01 |
| `type_diversity_factor` | 0.01 | 类型多样性惩罚权重 | 若某类偏少，升至 0.02–0.05 |
| `router_aux_loss_factor` | 0.02 | 负载均衡损失权重 | 若要更自由专业化，降至 0.01 |
| `num_experts_per_tok` | 2 | 每 token 激活几个专家 | 通常不改 |

---

## 8. 故障排除

### 8.1 常见问题

**Q: Windows 下报 NCCL 错误**  
A: Windows 不支持 NCCL。`run_eval.py` 已自动降级为单卡模式。训练脚本使用 HuggingFace Trainer，单卡下无需分布式。

**Q: `gradient_checkpointing` 在 Phase-A/B 报错 "tensors does not require grad"**  
A: Phase-A/B 大量参数被冻结，不兼容梯度检查点。解决方案：
- `h20` profile 启用 `gradient_checkpointing=True`，因为 Phase-C 占总步数 60%，收益大于前两阶段的轻微效率损失
- 或使用 `standard`/`dev` profile（不启用梯度检查点）

**Q: flash_attention_2 导入失败**  
A: 模型会自动降级为 eager attention，不影响正确性。若需要 flash-attn：
```bash
pip install flash-attn --no-build-isolation
```
仅 Linux + Ampere/Hopper GPU 支持。

**Q: OOM (显存不足)**  
A: 降低以下参数：
1. `--micro_batch 2` （优先）
2. `--max_length 256` 或 `128`
3. 使用 `--profile dev`

**Q: Electricity 数据集评估特别慢**  
A: Electricity 有 321 列，测试集很大。可减小 `--eval_batch_size 8` 或跳过：
```bash
python run_typemoe_production.py --datasets ETTh1 ETTh2 weather
```

### 8.2 TensorBoard 查看训练曲线

```bash
tensorboard --logdir logs/typemoe_h20/tb_logs
```

### 8.3 验证模型完整性

```python
from time_moe.models.modeling_time_moe import TimeMoeForPrediction
model = TimeMoeForPrediction.from_pretrained("logs/typemoe_h20")
print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")
print(f"路由模式: {model.config.router_mode}")
print(f"专家数: {model.config.num_experts}")
```

---

## 附录 A: 预期评估结果参考

### A.1 原始 TimeMoE-50M 基线 (0 步训练)

| 数据集 | pred_len=96 MSE |
|---|---|
| ETTh1 | 0.3577 |

### A.2 最小测试 (100 步)

| 数据集 | pred_len=96 MSE | pred_len=96 MAE |
|---|---|---|
| ETTh1 | 0.3603 | 0.3895 |

### A.3 生产训练预期 (5000 步)

| 数据集 | pred_len=96 | pred_len=192 | pred_len=336 | pred_len=720 |
|---|---|---|---|---|
| ETTh1 | ~0.32–0.34 | ~0.37–0.40 | ~0.40–0.44 | ~0.44–0.50 |
| ETTh2 | ~0.28–0.30 | ~0.34–0.37 | ~0.38–0.42 | ~0.42–0.48 |

> 注：以上预期值基于训练收敛假设，实际结果可能因超参数与数据而异。
