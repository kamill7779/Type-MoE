#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Type-MoE 最小端到端训练 + 评估流水线
=====================================
1. 数据准备: ETTh1.csv → JSON 训练格式
2. 训练: Type-MoE (typed routing + 异构专家 + 分阶段冻结)
3. 路由统计: 输出各层专家选择比例
4. 评估: 计算 MSE / MAE / WAPE

用法:
    python run_typemoe_minimal_test.py
"""
import os
import sys
import json
import time
import logging
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ── 日志配置 ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TypeMoE-Test")

# ── 路径配置 ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "TimeMoE-50M")
CONFIG_OVERRIDE = os.path.join(BASE_DIR, "configs", "typed_experts", "minimal_test.yaml")
CSV_PATH = os.path.join("E:/Project/Autoformer/dataset/ETT-small/ETTh1.csv")
TRAIN_DATA_DIR = os.path.join(BASE_DIR, "data_prepared")
TRAIN_DATA_PATH = os.path.join(TRAIN_DATA_DIR, "etth1_train.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "logs", "typemoe_minimal_test")

# ── 训练超参 (最小测试) ──
TRAIN_STEPS = 100           # 仅 100 步: 验证流程通畅
MAX_LENGTH = 128            # 较短窗口，节省显存
MICRO_BATCH = 2             # RTX 5060 显存有限
GLOBAL_BATCH = 4            # 小 batch
PRECISION = "bf16"          # bfloat16
LEARNING_RATE = 5e-4        # 略大 lr，100 步需要更快的收敛信号
MIN_LEARNING_RATE = 1e-5
PHASE_A_END = 20            # Phase-A: 仅 gate 可训 (步骤 0-19)
PHASE_B_END = 50            # Phase-B: gate + 新专家 (步骤 20-49)
                            # Phase-C: 全参数 (步骤 50+)

# ── 评估参数 ──
EVAL_PRED_LEN = 96
EVAL_CTX_LEN = 512
EVAL_BATCH_SIZE = 16


# =====================================================================
#  Step 1 : 数据准备 – ETTh1.csv → JSON (多序列)
# =====================================================================
def prepare_training_data():
    """把 ETTh1.csv 的 7 个特征列转换为 JSON 训练数据
    每列作为一条独立的时间序列，取前 60% 作为训练集"""
    if os.path.exists(TRAIN_DATA_PATH):
        log.info(f"训练数据已存在: {TRAIN_DATA_PATH}，跳过准备步骤")
        return

    log.info(f"读取 CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    cols = [c for c in df.columns if c != "date"]
    n = len(df)
    train_end = int(n * 0.6)  # ~10452 points
    log.info(f"总行数={n}, 使用前 {train_end} 行 (60%) 的 {len(cols)} 列作为训练序列")

    sequences = []
    for col in cols:
        seq = df[col].values[:train_end].astype(float).tolist()
        sequences.append(seq)

    os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
    with open(TRAIN_DATA_PATH, "w") as f:
        json.dump(sequences, f)
    log.info(f"训练数据已保存: {TRAIN_DATA_PATH} ({len(sequences)} 条序列, 每条 {train_end} 点)")


# =====================================================================
#  Step 2 : 训练 – 使用项目的 TimeMoeRunner
# =====================================================================
def train_model():
    """使用 Type-MoE 配置训练模型"""
    from time_moe.runner import TimeMoeRunner

    log.info("=" * 60)
    log.info("  Step 2: 开始 Type-MoE 训练")
    log.info(f"  模型基座: {MODEL_PATH}")
    log.info(f"  配置覆盖: {CONFIG_OVERRIDE}")
    log.info(f"  训练步数: {TRAIN_STEPS}")
    log.info(f"  窗口长度: {MAX_LENGTH}")
    log.info(f"  批大小: micro={MICRO_BATCH}, global={GLOBAL_BATCH}")
    log.info(f"  精度: {PRECISION}")
    log.info(f"  冻结策略: phased (A→{PHASE_A_END}, B→{PHASE_B_END})")
    log.info("=" * 60)

    runner = TimeMoeRunner(
        model_path=MODEL_PATH,
        output_path=OUTPUT_DIR,
        seed=42,
    )

    t0 = time.time()
    model = runner.train_model(
        from_scratch=False,
        data_path=TRAIN_DATA_PATH,
        max_length=MAX_LENGTH,
        stride=MAX_LENGTH,
        normalization_method="zero",
        attn_implementation="eager",
        micro_batch_size=MICRO_BATCH,
        global_batch_size=GLOBAL_BATCH,
        train_steps=TRAIN_STEPS,
        precision=PRECISION,
        learning_rate=LEARNING_RATE,
        min_learning_rate=MIN_LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        weight_decay=0.1,
        gradient_checkpointing=False,
        logging_steps=10,
        save_strategy="no",
        evaluation_strategy="no",
        max_grad_norm=1.0,
        dataloader_num_workers=0,       # Windows 兼容
        model_config_override=CONFIG_OVERRIDE,
        # Phased freeze 参数
        phase_a_end=PHASE_A_END,
        phase_b_end=PHASE_B_END,
    )
    elapsed = time.time() - t0
    log.info(f"训练完成! 用时 {elapsed:.1f}s, 模型已保存至: {OUTPUT_DIR}")
    return model


# =====================================================================
#  Step 3 : 路由统计 – 提取各层专家选择比例
# =====================================================================
def collect_routing_stats(model, device="cuda:0"):
    """对训练数据做一轮 forward pass，收集每层的路由统计信息"""
    log.info("=" * 60)
    log.info("  Step 3: 收集专家路由统计")
    log.info("=" * 60)

    model.eval()
    model.to(device)

    # 创建一小批测试数据进行 forward
    # 读取训练数据的前几条序列
    with open(TRAIN_DATA_PATH, "r") as f:
        sequences = json.load(f)

    config = model.config

    # 收集路由统计: layer_idx -> expert_idx -> count
    routing_stats = defaultdict(lambda: defaultdict(int))

    with torch.no_grad():
        for seq_i, seq in enumerate(sequences[:3]):  # 前3条序列足够
            # 创建多个窗口
            seq = np.array(seq, dtype=np.float32)
            for offset in range(0, len(seq) - MAX_LENGTH, MAX_LENGTH):
                window = seq[offset: offset + MAX_LENGTH]
                input_tensor = torch.tensor(window, dtype=torch.bfloat16).unsqueeze(0).to(device)

                # Forward pass (不需要 labels，只看路由)
                _ = model(input_ids=input_tensor)

                # 提取各层的路由信息
                for layer_idx, layer in enumerate(model.model.layers):
                    ffn = getattr(layer, "ffn_layer", None)
                    if ffn is None:
                        continue
                    last_routing = getattr(ffn, "_last_routing", None)
                    if not last_routing:
                        continue
                    selected = last_routing.get("selected_experts")
                    if selected is None:
                        continue
                    for expert_idx in selected.reshape(-1).tolist():
                        routing_stats[layer_idx][int(expert_idx)] += 1

            if seq_i == 0:
                # 只打一次日志确认 forward 正常
                log.info(f"序列 {seq_i}: forward pass 正常, 收集到 {len(routing_stats)} 层的路由数据")

    # 格式化输出
    expert_types = getattr(config, "expert_types", [])
    expert_type_map = getattr(config, "expert_type_map", [])
    custom_specs = getattr(config, "custom_expert_specs", [])
    num_experts = getattr(config, "num_experts", 8)

    # 构建专家名称
    expert_names = []
    for i in range(num_experts):
        spec = custom_specs[i] if i < len(custom_specs) else {}
        name = spec.get("name", f"expert_{i}") if isinstance(spec, dict) else f"expert_{i}"
        type_name = expert_types[expert_type_map[i]] if i < len(expert_type_map) and expert_type_map[i] < len(expert_types) else "?"
        expert_names.append(f"E{i}:{name}({type_name})")

    results = {}
    log.info("")
    log.info("┌─────────────────────────────────────────────────────────┐")
    log.info("│            专家路由选择比例 (Expert Routing Ratios)       │")
    log.info("├─────────────────────────────────────────────────────────┤")

    for layer_idx in sorted(routing_stats.keys()):
        layer_data = routing_stats[layer_idx]
        total = sum(layer_data.values())
        ratios = {}
        for e_idx in range(num_experts):
            count = layer_data.get(e_idx, 0)
            ratio = count / max(total, 1)
            ratios[expert_names[e_idx]] = {"count": count, "fraction": round(ratio, 4)}
        results[f"layer_{layer_idx}"] = ratios

        # 打印每层统计 (紧凑格式)
        ratio_strs = []
        for e_idx in range(num_experts):
            r = layer_data.get(e_idx, 0) / max(total, 1)
            ratio_strs.append(f"E{e_idx}:{r:.1%}")
        log.info(f"│ Layer {layer_idx:2d}: {' '.join(ratio_strs)}")

    log.info("└─────────────────────────────────────────────────────────┘")

    # 汇总: 各类型专家的总选择比例
    type_totals = defaultdict(int)
    grand_total = 0
    for layer_data in routing_stats.values():
        for e_idx, count in layer_data.items():
            type_id = expert_type_map[e_idx] if e_idx < len(expert_type_map) else -1
            type_name = expert_types[type_id] if 0 <= type_id < len(expert_types) else "unknown"
            type_totals[type_name] += count
            grand_total += count

    log.info("")
    log.info("按类型汇总:")
    for type_name in expert_types:
        ratio = type_totals[type_name] / max(grand_total, 1)
        log.info(f"  {type_name:>10}: {ratio:.1%} ({type_totals[type_name]}/{grand_total})")

    # 保存到 JSON
    routing_output = {
        "expert_names": expert_names,
        "expert_types": expert_types,
        "expert_type_map": expert_type_map,
        "per_layer": results,
        "type_summary": {k: round(v / max(grand_total, 1), 4) for k, v in type_totals.items()},
    }
    routing_path = os.path.join(OUTPUT_DIR, "routing_stats.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(routing_path, "w", encoding="utf-8") as f:
        json.dump(routing_output, f, indent=2, ensure_ascii=False)
    log.info(f"路由统计已保存: {routing_path}")

    return routing_stats


# =====================================================================
#  Step 4 : 评估 – 在 ETTh1 测试集计算 MSE / MAE / WAPE
# =====================================================================
def evaluate_model(model, device="cuda:0"):
    """对 ETTh1 测试集进行自回归预测，计算 MSE/MAE/WAPE"""
    from time_moe.datasets.benchmark_dataset import BenchmarkEvalDataset

    log.info("=" * 60)
    log.info(f"  Step 4: 评估 (ETTh1, pred_len={EVAL_PRED_LEN}, ctx_len={EVAL_CTX_LEN})")
    log.info("=" * 60)

    model.eval()
    model.to(device)

    dataset = BenchmarkEvalDataset(
        CSV_PATH,
        context_length=EVAL_CTX_LEN,
        prediction_length=EVAL_PRED_LEN,
    )
    log.info(f"测试集大小: {len(dataset)} 样本")

    dataloader = DataLoader(
        dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    sum_se = 0.0       # squared error
    sum_ae = 0.0       # absolute error
    sum_abs_label = 0.0 # |label| for WAPE
    total_count = 0

    t0 = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch["inputs"].to(device).to(model.dtype)
            labels = batch["labels"].to(device)

            # 自回归生成
            outputs = model.generate(
                inputs=inputs,
                max_new_tokens=EVAL_PRED_LEN,
            )
            preds = outputs[:, -EVAL_PRED_LEN:]
            if len(preds.shape) > len(labels.shape):
                labels = labels.unsqueeze(-1)

            preds_f = preds.float()
            labels_f = labels.float()

            sum_se += ((preds_f - labels_f) ** 2).sum().item()
            sum_ae += (preds_f - labels_f).abs().sum().item()
            sum_abs_label += labels_f.abs().sum().item()
            total_count += preds_f.numel()

            if (batch_idx + 1) % 50 == 0:
                log.info(f"  评估进度: {batch_idx + 1}/{len(dataloader)} batches")

    elapsed = time.time() - t0

    mse = sum_se / max(total_count, 1)
    mae = sum_ae / max(total_count, 1)
    wape = sum_ae / max(sum_abs_label, 1e-9)

    log.info("")
    log.info("┌─────────────────────────────────────────────┐")
    log.info("│         评估结果 (ETTh1 pred_len=96)          │")
    log.info("├─────────────────────────────────────────────┤")
    log.info(f"│  MSE  = {mse:.6f}                          │")
    log.info(f"│  MAE  = {mae:.6f}                          │")
    log.info(f"│  WAPE = {wape:.6f}                          │")
    log.info(f"│  用时  = {elapsed:.1f}s ({total_count} 预测点)    │")
    log.info("└─────────────────────────────────────────────┘")

    # 保存结果
    eval_results = {
        "dataset": "ETTh1",
        "prediction_length": EVAL_PRED_LEN,
        "context_length": EVAL_CTX_LEN,
        "MSE": round(mse, 6),
        "MAE": round(mae, 6),
        "WAPE": round(wape, 6),
        "total_predictions": total_count,
        "eval_time_sec": round(elapsed, 1),
    }
    eval_path = os.path.join(OUTPUT_DIR, "eval_results.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    log.info(f"评估结果已保存: {eval_path}")

    return eval_results


# =====================================================================
#  Main
# =====================================================================
def main():
    log.info("=" * 60)
    log.info("  Type-MoE 最小端到端测试 (Minimal E2E Pipeline)")
    log.info("=" * 60)

    # 检查环境
    if not torch.cuda.is_available():
        log.error("CUDA 不可用! 此脚本需要 GPU")
        sys.exit(1)
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    log.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    log.info(f"PyTorch: {torch.__version__}")

    if not os.path.exists(MODEL_PATH):
        log.error(f"基座模型不存在: {MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(CSV_PATH):
        log.error(f"数据集不存在: {CSV_PATH}")
        sys.exit(1)

    total_start = time.time()

    # ── Step 1: 准备数据 ──
    log.info("")
    log.info("=" * 60)
    log.info("  Step 1: 准备训练数据")
    log.info("=" * 60)
    prepare_training_data()

    # ── Step 2: 训练 ──
    model = train_model()

    # ── Step 3: 路由统计 ──
    device = "cuda:0"
    collect_routing_stats(model, device=device)

    # ── Step 4: 评估 ──
    evaluate_model(model, device=device)

    total_time = time.time() - total_start
    log.info("")
    log.info("=" * 60)
    log.info(f"  全部完成! 总用时: {total_time:.1f}s")
    log.info(f"  输出目录: {OUTPUT_DIR}")
    log.info(f"    - routing_stats.json  (专家路由统计)")
    log.info(f"    - eval_results.json   (MSE/MAE/WAPE)")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
