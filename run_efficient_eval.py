#!/usr/bin/env python
"""
Efficient batch evaluation: load model ONCE then evaluate on multiple datasets/horizons.
Outputs MSE/MAE table for TimeMoE-50M on standard benchmarks.
"""
import os
import sys
import time
import json
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATASET_ROOT = r"E:\Project\Autoformer\dataset"
MODEL_PATH = "./TimeMoE-50M"
BATCH_SIZE = 32

DATASETS = {
    "ETTh1": os.path.join(DATASET_ROOT, "ETT-small", "ETTh1.csv"),
    "ETTh2": os.path.join(DATASET_ROOT, "ETT-small", "ETTh2.csv"),
    "ETTm1": os.path.join(DATASET_ROOT, "ETT-small", "ETTm1.csv"),
    "ETTm2": os.path.join(DATASET_ROOT, "ETT-small", "ETTm2.csv"),
    "Weather": os.path.join(DATASET_ROOT, "weather", "weather.csv"),
    "Electricity": os.path.join(DATASET_ROOT, "electricity", "electricity.csv"),
}

PRED_LENGTHS = [96, 192, 336, 720]


def get_context_length(pred_len):
    mapping = {96: 512, 192: 1024, 336: 2048, 720: 3072}
    return mapping.get(pred_len, pred_len * 4)


def load_model(model_path, device):
    """Load model once."""
    from time_moe.models.modeling_time_moe import TimeMoeForPrediction
    try:
        model = TimeMoeForPrediction.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype='auto',
        )
    except Exception:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype='auto',
            trust_remote_code=True,
        )
    model.eval()
    logging.info(f"Model loaded: {type(model).__name__}, dtype={model.dtype}, device={device}")
    return model


def evaluate_one(model, device, csv_path, context_length, prediction_length, batch_size):
    """Evaluate one dataset + prediction length, return (mse, mae)."""
    from time_moe.datasets.benchmark_dataset import BenchmarkEvalDataset

    dataset = BenchmarkEvalDataset(
        csv_path,
        context_length=context_length,
        prediction_length=prediction_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    total_mse = 0.0
    total_mae = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"  p={prediction_length}", leave=False):
            inputs = batch['inputs'].to(device).to(model.dtype)
            labels = batch['labels'].to(device)

            outputs = model.generate(
                inputs=inputs,
                max_new_tokens=prediction_length,
            )
            preds = outputs[:, -prediction_length:]

            if len(preds.shape) > len(labels.shape):
                labels = labels[..., None]

            preds_f = preds.float()
            labels_f = labels.float()

            total_mse += torch.sum((preds_f - labels_f) ** 2).item()
            total_mae += torch.sum(torch.abs(preds_f - labels_f)).item()

            n = 1
            for s in preds.shape:
                n *= s
            total_count += n

    mse = total_mse / total_count
    mae = total_mae / total_count
    return mse, mae


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    if device.startswith("cuda"):
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}, "
                     f"Memory: {torch.cuda.get_device_properties(0).total_mem//(1024**3)}GB")

    # Load model once
    model = load_model(MODEL_PATH, device)

    results = {}
    all_start = time.time()

    for ds_name, csv_path in DATASETS.items():
        if not os.path.exists(csv_path):
            logging.warning(f"Skipping {ds_name}: file not found")
            continue

        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*60}")

        for pred_len in PRED_LENGTHS:
            ctx_len = get_context_length(pred_len)
            t0 = time.time()
            try:
                mse, mae = evaluate_one(model, device, csv_path, ctx_len, pred_len, BATCH_SIZE)
                elapsed = time.time() - t0
                results[f"{ds_name}_p{pred_len}"] = {
                    "dataset": ds_name,
                    "pred_len": pred_len,
                    "context_len": ctx_len,
                    "mse": round(mse, 6),
                    "mae": round(mae, 6),
                    "time_s": round(elapsed, 1),
                }
                print(f"  pred_len={pred_len:>4}  ctx={ctx_len:>5}  "
                      f"MSE={mse:.6f}  MAE={mae:.6f}  ({elapsed:.1f}s)")
            except Exception as e:
                logging.error(f"  {ds_name} p={pred_len}: {e}")
                import traceback
                traceback.print_exc()

    total_time = time.time() - all_start

    # Summary table
    print("\n\n" + "=" * 72)
    print("  TimeMoE-50M Evaluation Results")
    print("=" * 72)
    print(f"  {'Dataset':<15} {'Pred':<6} {'Context':<8} {'MSE':<12} {'MAE':<12} {'Time':<8}")
    print(f"  {'-'*15} {'-'*6} {'-'*8} {'-'*12} {'-'*12} {'-'*8}")

    for key in sorted(results.keys()):
        r = results[key]
        print(f"  {r['dataset']:<15} {r['pred_len']:<6} {r['context_len']:<8} "
              f"{r['mse']:<12.6f} {r['mae']:<12.6f} {r['time_s']:<8.1f}")

    print(f"\n  Total time: {total_time:.1f}s")
    print("=" * 72)

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
