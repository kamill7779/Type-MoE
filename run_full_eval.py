#!/usr/bin/env python
"""
Full batch evaluation for TimeMoE-50M on standard benchmarks.
Loads model ONCE, evaluates on multiple datasets and prediction lengths.
All output written to eval_full_output.txt and eval_results.json.
"""
import os, sys, time, json, torch
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from time_moe.datasets.benchmark_dataset import BenchmarkEvalDataset
from torch.utils.data import DataLoader

DATASET_ROOT = r"E:\Project\Autoformer\dataset"
MODEL_PATH = "./TimeMoE-50M"
BATCH_SIZE = 32
OUTPUT_FILE = "eval_full_output.txt"
RESULTS_FILE = "eval_results.json"

DATASETS = {
    "ETTh1": os.path.join(DATASET_ROOT, "ETT-small", "ETTh1.csv"),
    "ETTh2": os.path.join(DATASET_ROOT, "ETT-small", "ETTh2.csv"),
    "ETTm1": os.path.join(DATASET_ROOT, "ETT-small", "ETTm1.csv"),
    "ETTm2": os.path.join(DATASET_ROOT, "ETT-small", "ETTm2.csv"),
    "Weather": os.path.join(DATASET_ROOT, "weather", "weather.csv"),
}

PRED_LENGTHS = [96, 192, 336, 720]

CTX_MAP = {96: 512, 192: 1024, 336: 2048, 720: 3072}

log_f = open(OUTPUT_FILE, "w", buffering=1)
def log(msg):
    print(msg, flush=True)
    log_f.write(msg + "\n")
    log_f.flush()

def evaluate_one(model, device, csv_path, ctx_len, pred_len, batch_size):
    dataset = BenchmarkEvalDataset(csv_path, context_length=ctx_len, prediction_length=pred_len)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total_mse = 0.0
    total_mae = 0.0
    total_count = 0
    with torch.no_grad():
        for i, batch in enumerate(dl):
            inputs = batch['inputs'].to(device).to(model.dtype)
            labels = batch['labels'].to(device)
            outputs = model.generate(inputs=inputs, max_new_tokens=pred_len)
            preds = outputs[:, -pred_len:]
            if len(preds.shape) > len(labels.shape):
                labels = labels[..., None]
            pf = preds.float()
            lf = labels.float()
            total_mse += torch.sum((pf - lf) ** 2).item()
            total_mae += torch.sum(torch.abs(pf - lf)).item()
            n = 1
            for s in preds.shape:
                n *= s
            total_count += n
            if (i + 1) % 100 == 0:
                log(f"    batch {i+1}/{len(dl)}")
    mse = total_mse / total_count
    mae = total_mae / total_count
    return mse, mae


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")
    if device.startswith("cuda"):
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    log("Loading model...")
    from time_moe.models.modeling_time_moe import TimeMoeForPrediction
    model = TimeMoeForPrediction.from_pretrained(MODEL_PATH, device_map=device, torch_dtype="auto")
    model.eval()
    log(f"Model: {type(model).__name__}, dtype={model.dtype}")

    results = {}
    t_all = time.time()

    for ds_name, csv_path in DATASETS.items():
        if not os.path.exists(csv_path):
            log(f"SKIP {ds_name}: not found")
            continue
        for pred_len in PRED_LENGTHS:
            ctx_len = CTX_MAP.get(pred_len, pred_len * 4)
            log(f"\n--- {ds_name} pred={pred_len} ctx={ctx_len} ---")
            t0 = time.time()
            try:
                mse, mae = evaluate_one(model, device, csv_path, ctx_len, pred_len, BATCH_SIZE)
                elapsed = time.time() - t0
                results[f"{ds_name}_p{pred_len}"] = {
                    "dataset": ds_name, "pred_len": pred_len, "context_len": ctx_len,
                    "mse": round(mse, 6), "mae": round(mae, 6), "time_s": round(elapsed, 1)
                }
                log(f"  MSE={mse:.6f}  MAE={mae:.6f}  ({elapsed:.1f}s)")
                # Save intermediate results
                with open(RESULTS_FILE, "w") as f:
                    json.dump(results, f, indent=2)
            except Exception as e:
                log(f"  ERROR: {e}")
                import traceback
                traceback.print_exc(file=log_f)

    total_time = time.time() - t_all
    log(f"\n\n{'='*72}")
    log("  TimeMoE-50M Evaluation Results")
    log(f"{'='*72}")
    log(f"  {'Dataset':<12} {'Pred':<6} {'Ctx':<6} {'MSE':<12} {'MAE':<12} {'Time':<8}")
    log(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*12} {'-'*12} {'-'*8}")
    for key in sorted(results.keys()):
        r = results[key]
        log(f"  {r['dataset']:<12} {r['pred_len']:<6} {r['context_len']:<6} "
            f"{r['mse']:<12.6f} {r['mae']:<12.6f} {r['time_s']:<8.1f}")
    log(f"\n  Total: {total_time:.1f}s")
    log(f"{'='*72}")
    log_f.close()

if __name__ == "__main__":
    main()
