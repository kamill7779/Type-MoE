"""Quick single-dataset evaluation test to verify GPU inference works."""
import os, sys, time, torch
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Redirect all output to file
log_file = open("quick_eval_output.txt", "w", buffering=1)
def log(msg):
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

from time_moe.datasets.benchmark_dataset import BenchmarkEvalDataset
from torch.utils.data import DataLoader

device = "cuda:0" if torch.cuda.is_available() else "cpu"
log(f"Device: {device}")
if device.startswith("cuda"):
    log(f"GPU: {torch.cuda.get_device_name(0)}")

# Load model
log("Loading model...")
from time_moe.models.modeling_time_moe import TimeMoeForPrediction
model = TimeMoeForPrediction.from_pretrained("./TimeMoE-50M", device_map=device, torch_dtype="auto")
model.eval()
log(f"Model dtype: {model.dtype}")

# Dataset
csv_path = r"E:\Project\Autoformer\dataset\ETT-small\ETTh1.csv"
pred_len = 96
ctx_len = 512
dataset = BenchmarkEvalDataset(csv_path, context_length=ctx_len, prediction_length=pred_len)
log(f"Dataset size: {len(dataset)} samples")

dl = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

total_mse = 0.0
total_mae = 0.0
total_count = 0
t0 = time.time()

with torch.no_grad():
    for i, batch in enumerate(dl):
        inputs = batch['inputs'].to(device).to(model.dtype)
        labels = batch['labels'].to(device)
        outputs = model.generate(inputs=inputs, max_new_tokens=pred_len)
        preds = outputs[:, -pred_len:]
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
        if (i + 1) % 50 == 0:
            log(f"  Batch {i+1}/{len(dl)}...")

elapsed = time.time() - t0
mse = total_mse / total_count
mae = total_mae / total_count
log(f"\nETTh1 pred_len={pred_len}: MSE={mse:.6f}  MAE={mae:.6f}  ({elapsed:.1f}s)")
log_file.close()
