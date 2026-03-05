#!/usr/bin/env python
"""
Batch evaluation script for TimeMoE-50M on standard benchmark datasets.
Runs run_eval.py across multiple datasets and prediction lengths.
"""
import subprocess
import sys
import os
import json
import re
import time

PYTHON = sys.executable
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_EVAL = os.path.join(PROJECT_DIR, "run_eval.py")
MODEL_PATH = os.path.join(PROJECT_DIR, "TimeMoE-50M")
DATASET_ROOT = r"E:\Project\Autoformer\dataset"

# Datasets to evaluate
DATASETS = {
    "ETTh1": os.path.join(DATASET_ROOT, "ETT-small", "ETTh1.csv"),
    "ETTh2": os.path.join(DATASET_ROOT, "ETT-small", "ETTh2.csv"),
    "ETTm1": os.path.join(DATASET_ROOT, "ETT-small", "ETTm1.csv"),
    "ETTm2": os.path.join(DATASET_ROOT, "ETT-small", "ETTm2.csv"),
    "weather": os.path.join(DATASET_ROOT, "weather", "weather.csv"),
    "electricity": os.path.join(DATASET_ROOT, "electricity", "electricity.csv"),
}

# Prediction lengths: standard benchmark horizons
PREDICTION_LENGTHS = [96, 192, 336, 720]

# Batch size (reduce if OOM)
BATCH_SIZE = 16


def run_one(dataset_name, csv_path, pred_len):
    """Run evaluation for one dataset + prediction length, return (mse, mae) or None."""
    # Context length follows the same default logic as run_eval.py
    if pred_len == 96:
        ctx_len = 512
    elif pred_len == 192:
        ctx_len = 1024
    elif pred_len == 336:
        ctx_len = 2048
    elif pred_len == 720:
        ctx_len = 3072
    else:
        ctx_len = pred_len * 4

    cmd = [
        PYTHON, RUN_EVAL,
        "-m", MODEL_PATH,
        "-d", csv_path,
        "-b", str(BATCH_SIZE),
        "-c", str(ctx_len),
        "-p", str(pred_len),
    ]

    print(f"\n{'='*70}")
    print(f"  Dataset: {dataset_name}  |  Pred Length: {pred_len}  |  Context: {ctx_len}")
    print(f"{'='*70}")
    sys.stdout.flush()

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=PROJECT_DIR,
        )
        elapsed = time.time() - t0
        output = result.stdout + result.stderr

        # Parse metrics from output: look for {... 'mse': tensor(...), 'mae': tensor(...) ...}
        # The output format is: 0 - {'mse': tensor(value), 'mae': tensor(value)}
        mse_match = re.search(r"'mse':\s*(?:tensor\()?([0-9.eE+-]+)", output)
        mae_match = re.search(r"'mae':\s*(?:tensor\()?([0-9.eE+-]+)", output)

        if mse_match and mae_match:
            mse = float(mse_match.group(1))
            mae = float(mae_match.group(1))
            print(f"  Result: MSE={mse:.6f}  MAE={mae:.6f}  ({elapsed:.1f}s)")
            return mse, mae
        else:
            print(f"  [WARN] Could not parse metrics from output ({elapsed:.1f}s)")
            # Print last 15 lines for debugging
            lines = output.strip().split('\n')
            for line in lines[-15:]:
                print(f"    | {line}")
            return None
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] Evaluation took > 600s, skipping")
        return None
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


def main():
    print("=" * 70)
    print("  TimeMoE-50M Benchmark Evaluation")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Datasets: {list(DATASETS.keys())}")
    print(f"  Prediction Lengths: {PREDICTION_LENGTHS}")
    print("=" * 70)

    results = {}
    total_start = time.time()

    for ds_name, csv_path in DATASETS.items():
        if not os.path.exists(csv_path):
            print(f"\n  [SKIP] {ds_name}: file not found at {csv_path}")
            continue

        for pred_len in PREDICTION_LENGTHS:
            key = f"{ds_name}_p{pred_len}"
            metrics = run_one(ds_name, csv_path, pred_len)
            if metrics:
                results[key] = {"dataset": ds_name, "pred_len": pred_len, "mse": metrics[0], "mae": metrics[1]}

    total_elapsed = time.time() - total_start

    # Print summary table
    print("\n\n")
    print("=" * 70)
    print("  EVALUATION RESULTS SUMMARY  (TimeMoE-50M)")
    print("=" * 70)
    print(f"  {'Dataset':<15} {'Pred Len':<10} {'MSE':<12} {'MAE':<12}")
    print(f"  {'-'*15} {'-'*10} {'-'*12} {'-'*12}")

    for key in sorted(results.keys()):
        r = results[key]
        print(f"  {r['dataset']:<15} {r['pred_len']:<10} {r['mse']:<12.6f} {r['mae']:<12.6f}")

    print(f"\n  Total time: {total_elapsed:.1f}s")

    # Save results to JSON
    out_path = os.path.join(PROJECT_DIR, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
