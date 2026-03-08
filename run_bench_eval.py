#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Type-MoE 多数据集基准评估脚本
================================
加载训练好的 Type-MoE 模型 (logs/typemoe_2000step_multisrc) 以及
原始 TimeMoE-50M 基线模型，在 ETTh1/h2、ETTm1/m2、Weather 数据集
的 pred_len=96/192/336/720 上分别评估，并输出对比表格。

用法:
    python run_bench_eval.py [--mode both|typemoe|baseline]

    --mode both      (默认) 评估 Type-MoE 和 TimeMoE-50M 基线两者
    --mode typemoe   仅评估 Type-MoE 训练模型
    --mode baseline  仅评估原始 TimeMoE-50M 基线
"""
import os
import sys
import json
import time
import logging
import argparse
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

# ── 日志 ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("BenchEval")

# ── 路径 ──
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
TYPEMOE_DIR     = os.path.join(BASE_DIR, "logs", "typemoe_2000step_multisrc")  # 训练模型
BASELINE_DIR    = os.path.join(BASE_DIR, "TimeMoE-50M")                         # 原始基线

# ── 数据集路径 ──
DATASETS = {
    "ETTh1":   os.path.join(BASE_DIR, "data", "ETT-small", "ETTh1.csv"),
    "ETTh2":   os.path.join(BASE_DIR, "data", "ETT-small", "ETTh2.csv"),
    "ETTm1":   os.path.join(BASE_DIR, "data", "ETT-small", "ETTm1.csv"),
    "ETTm2":   os.path.join(BASE_DIR, "data", "ETT-small", "ETTm2.csv"),
    "Weather": os.path.join(BASE_DIR, "data", "weather.csv"),
}

# ── 评估参数 ──
PRED_LENS        = [96, 192, 336, 720]
CTX_LEN          = 512
EVAL_BATCH_SIZE  = 8       # 保守显存用量
DEFAULT_MAX_SAMP = 2000    # 每个(数据集, pred_len)组合最多评估的样本数，None=全量

# ── Table 2 中 Time-MoE 列的参考值 (仅供展示比较) ──
# 来源: 论文 Table 2, "Time-MoE" 列
TABLE2_TIMEMOE = {
    "ETTh1": {
        96:  (0.3604, 0.3960),
        192: (0.4076, 0.4197),
        336: (0.4219, 0.4323),
        720: (0.4495, 0.4540),
    },
    "ETTh2": {
        96:  (0.3698, 0.3960),
        192: (0.4458, 0.4469),
        336: (0.4744, 0.4689),
        720: (0.4701, 0.4803),
    },
    "ETTm1": {
        96:  (0.3004, 0.3439),
        192: (0.3453, 0.3726),
        336: (0.3746, 0.3918),
        720: (0.4323, 0.4378),
    },
    "ETTm2": {
        96:  (0.1574, 0.2534),
        192: (0.2129, 0.2961),
        336: (0.2639, 0.3280),
        720: (0.3561, 0.3998),
    },
    "Weather": {
        96:  (0.2110, 0.2774),
        192: (0.2600, 0.3065),
        336: (0.3158, 0.3452),
        720: (0.3878, 0.3956),
    },
}


# =====================================================================
#  模型加载
# =====================================================================
def load_model(model_dir: str, label: str = ""):
    """载入 TimeMoE 模型 (支持原始或 Type-MoE 版本)"""
    log.info(f"正在加载模型: {label} <- {model_dir}")

    # 确定是否有 config.json 中的自定义架构
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        log.error(f"config.json 不存在: {config_path}")
        sys.exit(1)

    # 先尝试用本地的 configuration/modeling 类
    try:
        sys.path.insert(0, BASE_DIR)
        from time_moe.models.configuration_time_moe import TimeMoeConfig
        from time_moe.models.modeling_time_moe import TimeMoeForPrediction

        config = TimeMoeConfig.from_pretrained(model_dir)
        model = TimeMoeForPrediction.from_pretrained(
            model_dir,
            config=config,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
    except Exception as e:
        log.warning(f"本地类加载失败 ({e}), 尝试原始 TimeMoE-50M 类...")
        # fallback: 使用 TimeMoE-50M 目录内的类 (原始基线)
        orig_dir = BASELINE_DIR
        sys.path.insert(0, orig_dir)
        import importlib.util

        def _load_from_dir(py_file, mod_name):
            spec = importlib.util.spec_from_file_location(mod_name, py_file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        cfg_mod = _load_from_dir(os.path.join(orig_dir, "configuration_time_moe.py"), "_orig_cfg")
        mdl_mod = _load_from_dir(os.path.join(orig_dir, "modeling_time_moe.py"), "_orig_mdl")

        config = cfg_mod.TimeMoeConfig.from_pretrained(model_dir)
        model = mdl_mod.TimeMoeForPrediction.from_pretrained(
            model_dir,
            config=config,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"  参数量: {total_params:.1f}M")
    return model


# =====================================================================
#  单次评估
# =====================================================================
def evaluate_one(model, csv_path: str, pred_len: int, ctx_len: int,
                 batch_size: int = 8, device: str = "cuda:0",
                 max_samples: int = None) -> dict:
    """在指定 CSV 数据集上以给定 pred_len 进行评估，返回 MSE / MAE / WAPE
    
    max_samples: 若不为 None，则对测试集进行均匀采样，限制最多评估样本数
    """
    from time_moe.datasets.benchmark_dataset import BenchmarkEvalDataset
    from torch.utils.data import Subset

    dataset = BenchmarkEvalDataset(csv_path, context_length=ctx_len, prediction_length=pred_len)
    if len(dataset) == 0:
        log.warning(f"数据集为空: {csv_path} pred_len={pred_len}")
        return {"MSE": float("nan"), "MAE": float("nan"), "WAPE": float("nan")}

    total_samples = len(dataset)
    if max_samples is not None and total_samples > max_samples:
        # 均匀采样 max_samples 个下标
        indices = np.linspace(0, total_samples - 1, max_samples, dtype=int)
        dataset = Subset(dataset, indices.tolist())
        log.info(f"    子集采样: {total_samples} -> {max_samples} 样本")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, drop_last=False)

    model.eval()
    model.to(device)

    sum_se = 0.0
    sum_ae = 0.0
    sum_abs_label = 0.0
    total_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch["inputs"].to(device).to(model.dtype)
            labels = batch["labels"].to(device)

            outputs = model.generate(inputs=inputs, max_new_tokens=pred_len)
            preds = outputs[:, -pred_len:]

            if len(preds.shape) > len(labels.shape):
                labels = labels.unsqueeze(-1)

            preds_f  = preds.float()
            labels_f = labels.float()

            sum_se         += ((preds_f - labels_f) ** 2).sum().item()
            sum_ae         += (preds_f - labels_f).abs().sum().item()
            sum_abs_label  += labels_f.abs().sum().item()
            total_count    += preds_f.numel()

            if (batch_idx + 1) % 100 == 0:
                log.info(f"    进度 {batch_idx + 1}/{len(dataloader)}")

    mse  = sum_se / max(total_count, 1)
    mae  = sum_ae / max(total_count, 1)
    wape = sum_ae / max(sum_abs_label, 1e-9)
    return {"MSE": round(mse, 4), "MAE": round(mae, 4), "WAPE": round(wape, 4)}


# =====================================================================
#  格式化输出
# =====================================================================
def print_comparison_table(results: dict):
    """
    results 结构:
      results[model_label][dataset][pred_len] = {"MSE": x, "MAE": y}
    """
    model_labels = list(results.keys())
    dataset_names = list(DATASETS.keys())

    # ── 表头 ──
    header_parts = ["Dataset", "Pred"]
    for lbl in model_labels:
        header_parts += [f"{lbl} MSE", f"{lbl} MAE"]
    header_parts += ["Paper Time-MoE MSE", "Paper Time-MoE MAE"]

    sep = "-" * (len(" | ".join(header_parts)) + 10)

    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("  Type-MoE vs 基线 对比结果  (ctx_len=512)")
    lines.append("=" * 80)
    lines.append(" | ".join(header_parts))
    lines.append(sep)

    for ds in dataset_names:
        for pred_len in PRED_LENS:
            row = [ds, str(pred_len)]
            for lbl in model_labels:
                r = results.get(lbl, {}).get(ds, {}).get(pred_len, {})
                mse = r.get("MSE", "N/A")
                mae = r.get("MAE", "N/A")
                row += [f"{mse:.4f}" if isinstance(mse, float) else mse,
                        f"{mae:.4f}" if isinstance(mae, float) else mae]
            # paper ref
            ref = TABLE2_TIMEMOE.get(ds, {}).get(pred_len, (None, None))
            row += [f"{ref[0]:.4f}" if ref[0] is not None else "N/A",
                    f"{ref[1]:.4f}" if ref[1] is not None else "N/A"]
            lines.append(" | ".join(row))
        lines.append(sep)

    print("\n".join(lines))
    return lines


def save_markdown_table(results: dict, save_path: str):
    """将结果保存为 Markdown 格式表格"""
    model_labels = list(results.keys())
    dataset_names = list(DATASETS.keys())

    md_lines = []
    md_lines.append("# Type-MoE 多数据集基准对比")
    md_lines.append(f"\n评估设置: `ctx_len={CTX_LEN}`, `pred_lens={PRED_LENS}`\n")

    # 构建 markdown 表头
    header = ["Dataset", "Pred len"]
    for lbl in model_labels:
        header += [f"{lbl} MSE", f"{lbl} MAE"]
    header += ["Paper(Time-MoE) MSE", "Paper(Time-MoE) MAE"]
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for ds in dataset_names:
        for pred_len in PRED_LENS:
            row = [ds, str(pred_len)]
            for lbl in model_labels:
                r = results.get(lbl, {}).get(ds, {}).get(pred_len, {})
                mse = r.get("MSE", "N/A")
                mae = r.get("MAE", "N/A")
                row += [f"{mse:.4f}" if isinstance(mse, float) else str(mse),
                        f"{mae:.4f}" if isinstance(mae, float) else str(mae)]
            ref = TABLE2_TIMEMOE.get(ds, {}).get(pred_len, (None, None))
            row += [f"{ref[0]:.4f}" if ref[0] is not None else "N/A",
                    f"{ref[1]:.4f}" if ref[1] is not None else "N/A"]
            md_lines.append("| " + " | ".join(row) + " |")

    md_text = "\n".join(md_lines)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    log.info(f"Markdown 结果已保存: {save_path}")
    return md_text


# =====================================================================
#  Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["both", "typemoe", "baseline"], default="both",
                        help="评估哪个模型: both/typemoe/baseline")
    parser.add_argument("--pred_lens", type=int, nargs="+", default=PRED_LENS,
                        help="要评估的预测步长列表 (默认: 96 192 336 720)")
    parser.add_argument("--datasets", type=str, nargs="+", default=list(DATASETS.keys()),
                        help="要评估的数据集名称列表 (默认: 全部)")
    parser.add_argument("--ctx_len", type=int, default=CTX_LEN,
                        help=f"上下文长度 (默认: {CTX_LEN})")
    parser.add_argument("--batch_size", type=int, default=EVAL_BATCH_SIZE,
                        help=f"评估批大小 (默认: {EVAL_BATCH_SIZE})")
    parser.add_argument("--output_dir", type=str, default=os.path.join(BASE_DIR, "logs", "bench_eval"),
                        help="结果保存目录")
    parser.add_argument("--max_samples", type=int, default=DEFAULT_MAX_SAMP,
                        help=f"每个评估组合最大样本数，0=全量 (默认: {DEFAULT_MAX_SAMP})")
    args = parser.parse_args()
    if args.max_samples == 0:
        args.max_samples = None   # 0 表示全量

    if not torch.cuda.is_available():
        log.error("CUDA 不可用!")
        sys.exit(1)
    device = "cuda:0"
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    log.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 收集要评估的模型 ──
    models_to_eval = {}
    if args.mode in ("both", "typemoe"):
        if not os.path.exists(TYPEMOE_DIR):
            log.error(f"Type-MoE 模型目录不存在: {TYPEMOE_DIR}")
            sys.exit(1)
        models_to_eval["Type-MoE(2kstep)"] = TYPEMOE_DIR

    if args.mode in ("both", "baseline"):
        if not os.path.exists(BASELINE_DIR):
            log.error(f"TimeMoE-50M 基线目录不存在: {BASELINE_DIR}")
            sys.exit(1)
        models_to_eval["TimeMoE-50M(base)"] = BASELINE_DIR

    log.info(f"待评估模型: {list(models_to_eval.keys())}")
    log.info(f"数据集: {args.datasets}")
    log.info(f"pred_lens: {args.pred_lens}")
    log.info(f"ctx_len: {args.ctx_len}, batch_size: {args.batch_size}")

    # 结果存储: results[model_label][dataset][pred_len] = {MSE, MAE, WAPE}
    all_results = {}
    raw_records = []

    total_combos = len(models_to_eval) * len(args.datasets) * len(args.pred_lens)
    done = 0
    grand_t0 = time.time()

    for model_label, model_dir in models_to_eval.items():
        log.info("")
        log.info("=" * 70)
        log.info(f"  评估模型: {model_label}")
        log.info("=" * 70)

        model = load_model(model_dir, label=model_label)
        all_results[model_label] = {}

        for ds_name in args.datasets:
            csv_path = DATASETS.get(ds_name)
            if csv_path is None:
                log.warning(f"未知数据集: {ds_name}, 跳过")
                continue
            if not os.path.exists(csv_path):
                log.warning(f"数据集文件不存在: {csv_path}, 跳过")
                continue

            all_results[model_label][ds_name] = {}
            log.info(f"\n  数据集: {ds_name}")

            for pred_len in args.pred_lens:
                t0 = time.time()
                log.info(f"    pred_len={pred_len} ...")
                metrics = evaluate_one(
                    model, csv_path, pred_len,
                    ctx_len=args.ctx_len,
                    batch_size=args.batch_size,
                    device=device,
                    max_samples=args.max_samples,
                )
                elapsed = time.time() - t0
                all_results[model_label][ds_name][pred_len] = metrics

                done += 1
                log.info(f"    [{done}/{total_combos}] {ds_name} pred={pred_len:3d} "
                         f"MSE={metrics['MSE']:.4f} MAE={metrics['MAE']:.4f} "
                         f"WAPE={metrics['WAPE']:.4f}  ({elapsed:.0f}s)")

                raw_records.append({
                    "model": model_label,
                    "dataset": ds_name,
                    "pred_len": pred_len,
                    **metrics,
                })

        # 释放显存
        model.cpu()
        del model
        torch.cuda.empty_cache()

    # ── 保存原始结果 JSON ──
    json_path = os.path.join(args.output_dir, "bench_eval_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"records": raw_records, "by_model": all_results}, f, indent=2, ensure_ascii=False)
    log.info(f"\n原始结果已保存: {json_path}")

    # ── 打印对比表 ──
    print_comparison_table(all_results)

    # ── 保存 Markdown ──
    md_path = os.path.join(args.output_dir, "bench_eval_table.md")
    save_markdown_table(all_results, md_path)

    total_elapsed = time.time() - grand_t0
    log.info(f"\n全部评估完成! 总用时: {total_elapsed:.0f}s")
    log.info(f"结果目录: {args.output_dir}")


if __name__ == "__main__":
    main()
