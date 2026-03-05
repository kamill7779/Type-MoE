#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Offline typed-routing analysis script (plan §13.4).

Loads a Type-MoE checkpoint, runs forward passes on a dataset, and
reports per-layer routing statistics: expert utilisation, type coverage,
load balance, and diversity metrics.

Usage:
    python scripts/analyze_typed_routing.py \
        --model path/to/checkpoint \
        --data path/to/dataset.csv \
        --context_length 512 \
        --prediction_length 96 \
        --batch_size 16 \
        --output routing_analysis.json
"""
import argparse
import collections
import json
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_model(model_path: str, device: str):
    try:
        from time_moe.models.modeling_time_moe import TimeMoeForPrediction
        model = TimeMoeForPrediction.from_pretrained(
            model_path, device_map=device, torch_dtype="auto"
        )
    except Exception:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=device, torch_dtype="auto", trust_remote_code=True
        )
    model.eval()
    return model


def get_dataset(data_path: str, context_length: int, prediction_length: int):
    from time_moe.datasets.benchmark_dataset import BenchmarkEvalDataset, GeneralEvalDataset
    if data_path.endswith(".csv"):
        return BenchmarkEvalDataset(data_path, context_length=context_length, prediction_length=prediction_length)
    return GeneralEvalDataset(data_path, context_length=context_length, prediction_length=prediction_length)


def collect_routing(model, dataloader, device, max_batches: int = None):
    """Run forward passes and collect routing stats from every MoE layer."""
    # per_layer[layer_idx] = {expert_idx: count, ...}
    per_layer_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    # per_layer_weights[layer_idx] = {expert_idx: cumulative_weight, ...}
    per_layer_weights = collections.defaultdict(lambda: collections.defaultdict(float))
    total_tokens = 0

    try:
        layers = model.model.layers
    except AttributeError:
        logger.error("Cannot access model.model.layers — unsupported model structure")
        return per_layer_counts, per_layer_weights, total_tokens

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting routing stats")):
            if max_batches is not None and batch_idx >= max_batches:
                break

            inputs = batch["inputs"].to(device).to(model.dtype)
            _ = model(input_ids=inputs)

            for layer_idx, layer in enumerate(layers):
                ffn = getattr(layer, "ffn_layer", None)
                if ffn is None:
                    continue
                routing = getattr(ffn, "_last_routing", None)
                if not routing:
                    continue

                selected = routing.get("selected_experts")  # [B*L, k]
                weights = routing.get("routing_weights")  # [B*L, k]
                if selected is None:
                    continue

                total_tokens += selected.shape[0]

                for k_idx in range(selected.shape[-1]):
                    experts_k = selected[:, k_idx].tolist()
                    weights_k = weights[:, k_idx].tolist() if weights is not None else [1.0] * len(experts_k)
                    for eidx, w in zip(experts_k, weights_k):
                        per_layer_counts[layer_idx][eidx] += 1
                        per_layer_weights[layer_idx][eidx] += float(w)

    return per_layer_counts, per_layer_weights, total_tokens


def analyse(model, per_layer_counts, per_layer_weights, total_tokens):
    """Produce analysis report dict."""
    config = model.config
    num_experts = getattr(config, "num_experts", 0)
    expert_type_map = getattr(config, "expert_type_map", [])
    expert_types = getattr(config, "expert_types", [])

    report = {
        "config": {
            "num_experts": num_experts,
            "expert_types": expert_types,
            "expert_type_map": expert_type_map,
            "router_mode": getattr(config, "router_mode", "standard"),
            "top_k": getattr(config, "num_experts_per_tok", 2),
        },
        "total_tokens_processed": total_tokens,
        "layers": {},
    }

    for layer_idx in sorted(per_layer_counts.keys()):
        counts = per_layer_counts[layer_idx]
        weights = per_layer_weights[layer_idx]
        total = sum(counts.values())

        experts_info = {}
        fractions = []
        for eidx in range(num_experts):
            c = counts.get(eidx, 0)
            w = weights.get(eidx, 0.0)
            frac = c / max(total, 1)
            fractions.append(frac)
            type_name = "unknown"
            if eidx < len(expert_type_map) and expert_type_map[eidx] < len(expert_types):
                type_name = expert_types[expert_type_map[eidx]]
            experts_info[str(eidx)] = {
                "count": c,
                "fraction": round(frac, 6),
                "avg_weight": round(w / max(c, 1), 6),
                "type": type_name,
            }

        fractions_arr = np.array(fractions)
        utilisation_variance = float(np.var(fractions_arr))
        non_zero_experts = int(np.sum(fractions_arr > 0))

        # Type coverage: fraction of types that have at least one selected expert
        type_counts = collections.defaultdict(int)
        for eidx in range(num_experts):
            if counts.get(eidx, 0) > 0 and eidx < len(expert_type_map):
                tid = expert_type_map[eidx]
                type_counts[tid] += 1
        num_types = len(expert_types) if expert_types else 1
        type_coverage = len(type_counts) / max(num_types, 1)

        report["layers"][str(layer_idx)] = {
            "total_selections": total,
            "active_experts": non_zero_experts,
            "utilisation_variance": round(utilisation_variance, 8),
            "type_coverage": round(type_coverage, 4),
            "experts": experts_info,
        }

    return report


def main():
    parser = argparse.ArgumentParser("Type-MoE Routing Analysis")
    parser.add_argument("--model", "-m", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--data", "-d", type=str, required=True, help="Dataset path")
    parser.add_argument("--context_length", "-c", type=int, default=512)
    parser.add_argument("--prediction_length", "-p", type=int, default=96)
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--max_batches", type=int, default=None, help="Max batches to process (None=all)")
    parser.add_argument("--output", "-o", type=str, default="routing_analysis.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    model = load_model(args.model, device)
    dataset = get_dataset(args.data, args.context_length, args.prediction_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=False)

    per_layer_counts, per_layer_weights, total_tokens = collect_routing(
        model, dataloader, device, max_batches=args.max_batches
    )

    report = analyse(model, per_layer_counts, per_layer_weights, total_tokens)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Analysis saved to {args.output}")

    # Print summary
    print("\n=== Routing Analysis Summary ===")
    print(f"Total tokens: {total_tokens}")
    for lid, linfo in report["layers"].items():
        print(f"\nLayer {lid}: {linfo['active_experts']}/{report['config']['num_experts']} experts active, "
              f"var={linfo['utilisation_variance']:.6f}, type_coverage={linfo['type_coverage']:.2f}")
        for eidx, einfo in linfo["experts"].items():
            print(f"  Expert {eidx} ({einfo['type']}): {einfo['fraction']:.4f} "
                  f"(count={einfo['count']}, avg_w={einfo['avg_weight']:.4f})")


if __name__ == "__main__":
    main()
