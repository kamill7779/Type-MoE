#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Type-MoE 生产训练 + 全量评估流水线
====================================
面向 H20-96G (96 GB HBM3) 设计，~12 小时预算
也可在小显存 GPU 上运行（自动降配）

流程:
  1. 数据准备: 多数据集 CSV → JSON
  2. 训练: Type-MoE (typed routing + 异构专家 + 分阶段冻结)
  3. 路由统计: 输出各层专家选择比例
  4. 多数据集 × 多预测长度评估

用法:
  # H20-96G 生产运行 (默认配置)
  python run_typemoe_production.py

  # 自定义参数
  python run_typemoe_production.py --train_steps 2000 --max_length 256 --datasets ETTh1 ETTh2

  # 仅评估已训练模型
  python run_typemoe_production.py --eval_only --model_dir logs/typemoe_h20

  # 小显存 GPU 快速验证
  python run_typemoe_production.py --profile dev
"""
import os
import sys
import json
import time
import math
import logging
import warnings
import argparse
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ── 日志配置 ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("TypeMoE-Prod")

# ── 项目根目录 ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════
#  预设配置 Profile
# ═══════════════════════════════════════════════════════════
PROFILES = {
    # ── H20-96G 生产 (默认) ──
    "h20": {
        "train_steps": 5000,
        "max_length": 512,
        "stride": 256,           # 50% overlap → 更多训练样本
        "micro_batch": 16,
        "global_batch": 64,
        "learning_rate": 2e-4,
        "min_learning_rate": 1e-5,
        "warmup_steps": 200,
        "phase_a_end": 500,
        "phase_b_end": 2000,
        "precision": "bf16",
        "gradient_checkpointing": True,  # Phase-C 启用
        "attn_impl": "auto",    # H20 支持 flash_attention_2
        "eval_batch_size": 64,
        "num_workers": 4,
        "config_override": "configs/typed_experts/h20_production.yaml",
        "datasets": ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity"],
        "pred_lens": [96, 192, 336, 720],
    },
    # ── 开发机 (8-16 GB VRAM) ──
    "dev": {
        "train_steps": 500,
        "max_length": 256,
        "stride": 256,
        "micro_batch": 2,
        "global_batch": 8,
        "learning_rate": 5e-4,
        "min_learning_rate": 1e-5,
        "warmup_steps": 20,
        "phase_a_end": 50,
        "phase_b_end": 200,
        "precision": "bf16",
        "gradient_checkpointing": False,
        "attn_impl": "eager",
        "eval_batch_size": 16,
        "num_workers": 0,
        "config_override": "configs/typed_experts/h20_production.yaml",
        "datasets": ["ETTh1"],
        "pred_lens": [96],
    },
    # ── 标准调优 (24-48 GB VRAM) ──
    "standard": {
        "train_steps": 2000,
        "max_length": 512,
        "stride": 256,
        "micro_batch": 8,
        "global_batch": 32,
        "learning_rate": 3e-4,
        "min_learning_rate": 1e-5,
        "warmup_steps": 100,
        "phase_a_end": 200,
        "phase_b_end": 800,
        "precision": "bf16",
        "gradient_checkpointing": False,
        "attn_impl": "eager",
        "eval_batch_size": 32,
        "num_workers": 2,
        "config_override": "configs/typed_experts/h20_production.yaml",
        "datasets": ["ETTh1", "ETTh2", "weather"],
        "pred_lens": [96, 192, 336],
    },
}

# ═══════════════════════════════════════════════════════════
#  数据集注册表
# ═══════════════════════════════════════════════════════════
DATASET_REGISTRY = {
    "ETTh1": {"csv": "data/ETT-small/ETTh1.csv", "freq": "h", "cols": 7},
    "ETTh2": {"csv": "data/ETT-small/ETTh2.csv", "freq": "h", "cols": 7},
    "ETTm1": {"csv": "data/ETT-small/ETTm1.csv", "freq": "15min", "cols": 7},
    "ETTm2": {"csv": "data/ETT-small/ETTm2.csv", "freq": "15min", "cols": 7},
    "weather": {"csv": "data/weather.csv", "freq": "10min", "cols": 21},
    "electricity": {"csv": "data/electricity.csv", "freq": "h", "cols": 321},
}

# pred_len → ctx_len 映射 (与 run_eval.py 对齐)
CTX_LEN_MAP = {96: 512, 192: 1024, 336: 2048, 720: 3072}


# ═══════════════════════════════════════════════════════════
#  Step 1: 数据准备
# ═══════════════════════════════════════════════════════════
def prepare_training_data(dataset_names, output_dir, train_ratio=0.6):
    """将多个 CSV 数据集的训练部分合并为 JSON 格式"""
    json_path = os.path.join(output_dir, "train_data.json")
    if os.path.exists(json_path):
        log.info(f"训练数据已存在: {json_path}，跳过准备步骤")
        return json_path

    all_sequences = []
    for name in dataset_names:
        info = DATASET_REGISTRY.get(name)
        if info is None:
            log.warning(f"未知数据集 '{name}'，跳过")
            continue
        csv_path = os.path.join(BASE_DIR, info["csv"])
        if not os.path.exists(csv_path):
            log.warning(f"数据集文件不存在: {csv_path}，跳过")
            continue

        df = pd.read_csv(csv_path)
        cols = [c for c in df.columns if c.lower() not in ("date", "ot")]
        n = len(df)
        train_end = int(n * train_ratio)

        for col in cols:
            seq = df[col].values[:train_end].astype(float).tolist()
            all_sequences.append(seq)

        log.info(f"  {name}: {len(cols)} 列 × {train_end} 点 → {len(cols)} 条序列")

    os.makedirs(output_dir, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(all_sequences, f)

    total_points = sum(len(s) for s in all_sequences)
    log.info(f"训练数据已保存: {json_path}")
    log.info(f"  共 {len(all_sequences)} 条序列, "
             f"{total_points:,} 个时间点, "
             f"{os.path.getsize(json_path) / 1e6:.1f} MB")
    return json_path


# ═══════════════════════════════════════════════════════════
#  Step 2: 训练
# ═══════════════════════════════════════════════════════════
def train_model(cfg, model_path, output_dir, train_data_path):
    """使用 Type-MoE 配置训练模型"""
    from time_moe.runner import TimeMoeRunner

    log.info("=" * 70)
    log.info("  Step 2: Type-MoE 训练")
    log.info(f"  基座模型: {model_path}")
    log.info(f"  配置文件: {cfg['config_override']}")
    log.info(f"  训练步数: {cfg['train_steps']}")
    log.info(f"  窗口长度: {cfg['max_length']}, 步长: {cfg['stride']}")
    log.info(f"  批大小: micro={cfg['micro_batch']}, global={cfg['global_batch']}")
    log.info(f"  分阶段冻结: A→{cfg['phase_a_end']}, B→{cfg['phase_b_end']}")
    log.info(f"  精度: {cfg['precision']}, 注意力: {cfg['attn_impl']}")
    log.info("=" * 70)

    runner = TimeMoeRunner(
        model_path=model_path,
        output_path=output_dir,
        seed=42,
    )

    config_override_path = os.path.join(BASE_DIR, cfg["config_override"])

    t0 = time.time()
    model = runner.train_model(
        from_scratch=False,
        data_path=train_data_path,
        max_length=cfg["max_length"],
        stride=cfg["stride"],
        normalization_method="zero",
        attn_implementation=cfg["attn_impl"],
        micro_batch_size=cfg["micro_batch"],
        global_batch_size=cfg["global_batch"],
        train_steps=cfg["train_steps"],
        precision=cfg["precision"],
        learning_rate=cfg["learning_rate"],
        min_learning_rate=cfg["min_learning_rate"],
        lr_scheduler_type="cosine",
        warmup_steps=cfg["warmup_steps"],
        weight_decay=0.1,
        gradient_checkpointing=cfg["gradient_checkpointing"],
        logging_steps=max(1, cfg["train_steps"] // 100),  # ~100 条日志
        save_strategy="steps",
        save_steps=max(500, cfg["train_steps"] // 5),  # 每 1/5 保存
        save_total_limit=3,
        save_only_model=True,
        evaluation_strategy="no",
        max_grad_norm=1.0,
        dataloader_num_workers=cfg["num_workers"],
        model_config_override=config_override_path,
        phase_a_end=cfg["phase_a_end"],
        phase_b_end=cfg["phase_b_end"],
    )

    elapsed = time.time() - t0
    hours = elapsed / 3600
    log.info(f"训练完成! 用时 {elapsed:.0f}s ({hours:.2f}h)")
    log.info(f"  平均 {elapsed / cfg['train_steps']:.2f}s/步")
    return model


# ═══════════════════════════════════════════════════════════
#  Step 3: 路由统计
# ═══════════════════════════════════════════════════════════
def collect_routing_stats(model, train_data_path, output_dir, max_length=512,
                          device="cuda:0", max_sequences=5):
    """对训练数据做一轮 forward pass，收集每层的路由统计信息"""
    log.info("=" * 70)
    log.info("  Step 3: 收集专家路由统计")
    log.info("=" * 70)

    model.eval()
    model.to(device)

    with open(train_data_path, "r") as f:
        sequences = json.load(f)

    config = model.config
    routing_stats = defaultdict(lambda: defaultdict(int))

    num_seq = min(len(sequences), max_sequences)
    with torch.no_grad():
        for seq_i in range(num_seq):
            seq = np.array(sequences[seq_i], dtype=np.float32)
            for offset in range(0, len(seq) - max_length, max_length):
                window = seq[offset: offset + max_length]
                input_tensor = torch.tensor(window, dtype=torch.bfloat16).unsqueeze(0).to(device)
                _ = model(input_ids=input_tensor)

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

    # 格式化
    expert_types = getattr(config, "expert_types", [])
    expert_type_map = getattr(config, "expert_type_map", [])
    custom_specs = getattr(config, "custom_expert_specs", [])
    num_experts = getattr(config, "num_experts", 8)

    expert_names = []
    for i in range(num_experts):
        spec = custom_specs[i] if i < len(custom_specs) else {}
        name = spec.get("name", f"expert_{i}") if isinstance(spec, dict) else f"expert_{i}"
        type_name = expert_types[expert_type_map[i]] if i < len(expert_type_map) and expert_type_map[i] < len(expert_types) else "?"
        expert_names.append(f"E{i}:{name}({type_name})")

    # 按类型汇总
    type_totals = defaultdict(int)
    grand_total = 0
    for layer_data in routing_stats.values():
        for e_idx, count in layer_data.items():
            type_id = expert_type_map[e_idx] if e_idx < len(expert_type_map) else -1
            type_name = expert_types[type_id] if 0 <= type_id < len(expert_types) else "unknown"
            type_totals[type_name] += count
            grand_total += count

    log.info("按类型汇总路由比例:")
    for type_name in expert_types:
        ratio = type_totals[type_name] / max(grand_total, 1)
        log.info(f"  {type_name:>10}: {ratio:.1%} ({type_totals[type_name]:,}/{grand_total:,})")

    # 每层统计
    results = {}
    for layer_idx in sorted(routing_stats.keys()):
        layer_data = routing_stats[layer_idx]
        total = sum(layer_data.values())
        ratios = {}
        for e_idx in range(num_experts):
            count = layer_data.get(e_idx, 0)
            ratio = count / max(total, 1)
            ratios[expert_names[e_idx]] = {"count": count, "fraction": round(ratio, 4)}
        results[f"layer_{layer_idx}"] = ratios

    # 保存
    routing_output = {
        "timestamp": datetime.now().isoformat(),
        "expert_names": expert_names,
        "expert_types": expert_types,
        "expert_type_map": expert_type_map,
        "total_selections": grand_total,
        "type_summary": {k: round(v / max(grand_total, 1), 4) for k, v in type_totals.items()},
        "per_layer": results,
    }
    routing_path = os.path.join(output_dir, "routing_stats.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(routing_path, "w", encoding="utf-8") as f:
        json.dump(routing_output, f, indent=2, ensure_ascii=False)
    log.info(f"路由统计已保存: {routing_path}")
    return routing_output


# ═══════════════════════════════════════════════════════════
#  Step 4: 多数据集 × 多预测长度评估
# ═══════════════════════════════════════════════════════════
def evaluate_single(model, csv_path, pred_len, ctx_len, batch_size, device="cuda:0"):
    """对单个数据集 + 预测长度进行评估，返回 MSE/MAE/WAPE"""
    from time_moe.datasets.benchmark_dataset import BenchmarkEvalDataset

    model.eval()
    model.to(device)

    dataset = BenchmarkEvalDataset(csv_path, context_length=ctx_len, prediction_length=pred_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    sum_se = 0.0
    sum_ae = 0.0
    sum_abs_label = 0.0
    total_count = 0

    t0 = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch["inputs"].to(device).to(model.dtype)
            labels = batch["labels"].to(device)

            outputs = model.generate(inputs=inputs, max_new_tokens=pred_len)
            preds = outputs[:, -pred_len:]
            if len(preds.shape) > len(labels.shape):
                labels = labels.unsqueeze(-1)

            preds_f = preds.float()
            labels_f = labels.float()

            sum_se += ((preds_f - labels_f) ** 2).sum().item()
            sum_ae += (preds_f - labels_f).abs().sum().item()
            sum_abs_label += labels_f.abs().sum().item()
            total_count += preds_f.numel()

    elapsed = time.time() - t0
    mse = sum_se / max(total_count, 1)
    mae = sum_ae / max(total_count, 1)
    wape = sum_ae / max(sum_abs_label, 1e-9)

    return {
        "MSE": round(mse, 6),
        "MAE": round(mae, 6),
        "WAPE": round(wape, 6),
        "total_predictions": total_count,
        "eval_time_sec": round(elapsed, 1),
    }


def evaluate_all(model, cfg, output_dir, device="cuda:0"):
    """对所有数据集 × 预测长度进行评估"""
    log.info("=" * 70)
    log.info("  Step 4: 多数据集评估")
    log.info(f"  数据集: {cfg['datasets']}")
    log.info(f"  预测长度: {cfg['pred_lens']}")
    log.info("=" * 70)

    all_results = {}
    summary_rows = []

    total_eval_start = time.time()

    for ds_name in cfg["datasets"]:
        info = DATASET_REGISTRY.get(ds_name)
        if info is None:
            log.warning(f"未知数据集 '{ds_name}'，跳过评估")
            continue
        csv_path = os.path.join(BASE_DIR, info["csv"])
        if not os.path.exists(csv_path):
            log.warning(f"数据集文件不存在: {csv_path}，跳过评估")
            continue

        ds_results = {}
        for pred_len in cfg["pred_lens"]:
            ctx_len = CTX_LEN_MAP.get(pred_len, pred_len * 4)
            log.info(f"  评估 {ds_name} pred_len={pred_len} ctx_len={ctx_len} ...")

            try:
                result = evaluate_single(
                    model, csv_path, pred_len, ctx_len,
                    batch_size=cfg["eval_batch_size"],
                    device=device,
                )
                ds_results[str(pred_len)] = result
                summary_rows.append({
                    "dataset": ds_name,
                    "pred_len": pred_len,
                    "ctx_len": ctx_len,
                    **result,
                })
                log.info(f"    MSE={result['MSE']:.6f}  MAE={result['MAE']:.6f}  "
                         f"WAPE={result['WAPE']:.6f}  ({result['eval_time_sec']:.0f}s)")
            except Exception as e:
                log.error(f"    评估失败: {e}")
                ds_results[str(pred_len)] = {"error": str(e)}

        all_results[ds_name] = ds_results

    total_eval_time = time.time() - total_eval_start

    # 打印汇总表
    log.info("")
    log.info("=" * 70)
    log.info("  评估结果汇总")
    log.info("=" * 70)
    log.info(f"  {'Dataset':<15} {'pred_len':>8} {'MSE':>10} {'MAE':>10} {'WAPE':>10}")
    log.info("  " + "-" * 55)
    for row in summary_rows:
        if "error" not in row:
            log.info(f"  {row['dataset']:<15} {row['pred_len']:>8} "
                     f"{row['MSE']:>10.6f} {row['MAE']:>10.6f} {row['WAPE']:>10.6f}")
    log.info(f"  总评估用时: {total_eval_time:.0f}s ({total_eval_time/3600:.2f}h)")

    # 保存
    eval_output = {
        "timestamp": datetime.now().isoformat(),
        "datasets": cfg["datasets"],
        "pred_lens": cfg["pred_lens"],
        "total_eval_time_sec": round(total_eval_time, 1),
        "results": all_results,
        "summary": summary_rows,
    }
    eval_path = os.path.join(output_dir, "eval_results.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=2, ensure_ascii=False)
    log.info(f"评估结果已保存: {eval_path}")

    return eval_output


# ═══════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(
        description="Type-MoE 生产训练 + 评估流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # 基本参数
    parser.add_argument("--profile", type=str, default="h20",
                        choices=list(PROFILES.keys()),
                        help="预设配置 (default: h20)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="基座模型路径 (default: TimeMoE-50M)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录 (default: logs/typemoe_{profile})")
    parser.add_argument("--eval_only", action="store_true",
                        help="仅评估 (跳过训练)")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="--eval_only 时指定已训练模型路径")
    parser.add_argument("--skip_routing", action="store_true",
                        help="跳过路由统计收集")

    # 可覆盖 profile 的参数
    parser.add_argument("--train_steps", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--micro_batch", type=int, default=None)
    parser.add_argument("--global_batch", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--precision", type=str, default=None, choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--phase_a_end", type=int, default=None)
    parser.add_argument("--phase_b_end", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="训练+评估数据集列表")
    parser.add_argument("--pred_lens", nargs="+", type=int, default=None,
                        help="评估预测长度列表")

    return parser.parse_args()


def main():
    args = parse_args()

    # 加载 profile 并应用 CLI 覆盖
    cfg = PROFILES[args.profile].copy()
    overridable = [
        "train_steps", "max_length", "stride", "micro_batch", "global_batch",
        "learning_rate", "precision", "phase_a_end", "phase_b_end",
        "eval_batch_size", "datasets", "pred_lens",
    ]
    for key in overridable:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    # 路径配置
    model_path = args.model_path or os.path.join(BASE_DIR, "TimeMoE-50M")
    output_dir = args.output_dir or os.path.join(BASE_DIR, "logs", f"typemoe_{args.profile}")
    data_prep_dir = os.path.join(output_dir, "data_prepared")
    os.makedirs(output_dir, exist_ok=True)

    # 保存运行配置
    run_config = {
        "timestamp": datetime.now().isoformat(),
        "profile": args.profile,
        "config": cfg,
        "model_path": model_path,
        "output_dir": output_dir,
        "eval_only": args.eval_only,
    }
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    log.info("=" * 70)
    log.info(f"  Type-MoE 生产流水线 (profile={args.profile})")
    log.info("=" * 70)

    # 检查环境
    if not torch.cuda.is_available():
        log.error("CUDA 不可用!")
        sys.exit(1)
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    log.info(f"GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")
    log.info(f"PyTorch: {torch.__version__}")
    log.info(f"训练步数: {cfg['train_steps']}, 窗口: {cfg['max_length']}")
    log.info(f"数据集: {cfg['datasets']}")
    log.info(f"预测长度: {cfg['pred_lens']}")

    if not os.path.exists(model_path):
        log.error(f"基座模型不存在: {model_path}")
        sys.exit(1)

    device = "cuda:0"
    total_start = time.time()

    if args.eval_only:
        # ── 仅评估模式 ──
        eval_model_path = args.model_dir or output_dir
        log.info(f"仅评估模式，加载模型: {eval_model_path}")
        from time_moe.models.modeling_time_moe import TimeMoeForPrediction
        model = TimeMoeForPrediction.from_pretrained(
            eval_model_path, torch_dtype=torch.bfloat16, device_map=device
        )
    else:
        # ── Step 1: 准备数据 ──
        log.info("")
        log.info("=" * 70)
        log.info("  Step 1: 准备训练数据")
        log.info("=" * 70)
        train_data_path = prepare_training_data(cfg["datasets"], data_prep_dir)

        # ── Step 2: 训练 ──
        model = train_model(cfg, model_path, output_dir, train_data_path)

        # ── Step 3: 路由统计 ──
        if not args.skip_routing:
            collect_routing_stats(
                model, train_data_path, output_dir,
                max_length=cfg["max_length"], device=device,
            )

    # ── Step 4: 评估 ──
    evaluate_all(model, cfg, output_dir, device=device)

    total_time = time.time() - total_start
    hours = total_time / 3600

    log.info("")
    log.info("=" * 70)
    log.info(f"  全部完成! 总用时: {total_time:.0f}s ({hours:.2f}h)")
    log.info(f"  输出目录: {output_dir}")
    log.info(f"    - run_config.json     (运行配置)")
    log.info(f"    - routing_stats.json  (专家路由统计)")
    log.info(f"    - eval_results.json   (多数据集评估结果)")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
