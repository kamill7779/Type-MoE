#!/usr/bin/env python
# -*- coding:utf-8 _*-
import json
import os
import argparse
import collections
import numpy as np
import logging
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
from tqdm import tqdm

from transformers import AutoModelForCausalLM

from time_moe.datasets.benchmark_dataset import BenchmarkEvalDataset, GeneralEvalDataset


def setup_nccl(rank, world_size, master_addr='127.0.0.1', master_port=9899):
    dist.init_process_group("nccl", init_method='tcp://{}:{}'.format(master_addr, master_port), rank=rank,
                            world_size=world_size)


def count_num_tensor_elements(tensor):
    n = 1
    for s in tensor.shape:
        n = n * s
    return n


# ------------------ Metrics ------------------
class SumEvalMetric:
    def __init__(self, name, init_val: float = 0.0):
        self.name = name
        self.value = init_val

    def push(self, preds, labels, **kwargs):
        self.value += self._calculate(preds, labels, **kwargs)

    def _calculate(self, preds, labels, **kwargs):
        pass


class MSEMetric(SumEvalMetric):
    def _calculate(self, preds, labels, **kwargs):
        return torch.sum((preds - labels) ** 2)


class MAEMetric(SumEvalMetric):
    def _calculate(self, preds, labels, **kwargs):
        return torch.sum(torch.abs(preds - labels))


class TimeMoE:
    def __init__(self, model_path, device, context_length, prediction_length, **kwargs):
        try:
            from time_moe.models.modeling_time_moe import TimeMoeForPrediction
            model = TimeMoeForPrediction.from_pretrained(
                model_path,
                device_map=device,
                # attn_implementation='flash_attention_2',
                torch_dtype='auto',
            )
        except:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                # attn_implementation='flash_attention_2',
                torch_dtype='auto',
                trust_remote_code=True,
            )

        logging.info(f'>>> Model dtype: {model.dtype}; Attention:{model.config._attn_implementation}')

        self.model = model
        self.device = device
        self.prediction_length = prediction_length
        self.model.eval()

    def predict(self, batch):
        model = self.model
        device = self.device
        prediction_length = self.prediction_length

        outputs = model.generate(
            inputs=batch['inputs'].to(device).to(model.dtype),
            max_new_tokens=prediction_length,
        )
        preds = outputs[:, -prediction_length:]
        labels = batch['labels'].to(device)
        if len(preds.shape) > len(labels.shape):
            labels = labels[..., None]
        return preds, labels


def _collect_routing_stats_from_model(internal_model, routing_stats):
    """Collect per-layer routing statistics from _last_routing in each MoE layer."""
    try:
        layers = internal_model.model.layers
    except AttributeError:
        return
    for layer_idx, layer in enumerate(layers):
        ffn = getattr(layer, 'ffn_layer', None)
        if ffn is None:
            continue
        last_routing = getattr(ffn, '_last_routing', None)
        if not last_routing:
            continue
        selected = last_routing.get('selected_experts')
        if selected is None:
            continue
        # Count per-expert selections
        for expert_idx in selected.reshape(-1).tolist():
            routing_stats[layer_idx][expert_idx] = routing_stats[layer_idx].get(expert_idx, 0) + 1


def _format_routing_stats(routing_stats, model_wrapper):
    """Format routing stats into a JSON-serialisable dict."""
    try:
        config = model_wrapper.model.config
        expert_type_map = getattr(config, 'expert_type_map', [])
        expert_types = getattr(config, 'expert_types', [])
        num_experts = getattr(config, 'num_experts', 0)
    except AttributeError:
        expert_type_map = []
        expert_types = []
        num_experts = 0

    output = {
        'num_experts': num_experts,
        'expert_types': expert_types,
        'expert_type_map': expert_type_map,
        'per_layer': {},
    }
    for layer_idx in sorted(routing_stats.keys()):
        layer_data = routing_stats[layer_idx]
        total = sum(layer_data.values())
        expert_stats = {}
        for eidx in range(num_experts):
            count = layer_data.get(eidx, 0)
            expert_stats[str(eidx)] = {
                'count': count,
                'fraction': count / max(total, 1),
                'type': expert_types[expert_type_map[eidx]] if eidx < len(expert_type_map) and expert_type_map[eidx] < len(expert_types) else 'unknown',
            }
        output['per_layer'][str(layer_idx)] = {
            'total_selections': total,
            'experts': expert_stats,
        }
    return output


def evaluate(args):
    batch_size = args.batch_size
    context_length = args.context_length
    prediction_length = args.prediction_length

    master_addr = os.getenv('MASTER_ADDR', '127.0.0.1')
    master_port = os.getenv('MASTER_PORT', 9899)
    world_size = int(os.getenv('WORLD_SIZE') or 1)
    rank = int(os.getenv('RANK') or 0)
    local_rank = int(os.getenv('LOCAL_RANK') or 0)
    if torch.cuda.is_available():
        try:
            setup_nccl(rank=rank, world_size=world_size, master_addr=master_addr, master_port=master_port)
            device = f"cuda:{local_rank}"
            is_dist = True
        except Exception as e:
            print(f'Info: NCCL distributed init failed ({e}), using single-GPU mode.')
            device = f"cuda:{local_rank}"
            is_dist = False
    else:
        device = 'cpu'
        is_dist = False

    # evaluation
    metric_list = [
        MSEMetric(name='mse'),
        MAEMetric(name='mae'),
    ]

    model = TimeMoE(
        args.model,
        device,
        context_length=context_length,
        prediction_length=prediction_length
    )
    if args.data.endswith('.csv'):
        dataset = BenchmarkEvalDataset(
            args.data,
            context_length=context_length,
            prediction_length=prediction_length,
        )
    else:
        dataset = GeneralEvalDataset(
            args.data,
            context_length=context_length,
            prediction_length=prediction_length,
        )

    if torch.cuda.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset=dataset, shuffle=False)
    else:
        sampler = None
    test_dl = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
        drop_last=False,
    )

    acc_count = 0
    routing_stats = collections.defaultdict(lambda: collections.defaultdict(float))
    export_routing = getattr(args, 'export_routing_stats', False)

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_dl)):
            preds, labels = model.predict(batch)

            for metric in metric_list:
                metric.push(preds, labels)

            acc_count += count_num_tensor_elements(preds)

            # Collect routing stats if requested
            if export_routing:
                _collect_routing_stats_from_model(model.model, routing_stats)

    ret_metric = {}
    for metric in metric_list:
        ret_metric[metric.name] = metric.value / acc_count
    print(f'{rank} - {ret_metric}')

    metric_tensors = [metric.value for metric in metric_list] + [acc_count]
    if is_dist:
        stat_tensor = torch.tensor(metric_tensors).to(model.device)
        gathered_results = [torch.zeros_like(stat_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_results, stat_tensor)
        all_stat = torch.stack(gathered_results, dim=0).sum(dim=0)
    else:
        all_stat = metric_tensors

    if rank == 0:
        item = {
            'model': args.model,
            'data': args.data,
            'context_length': args.context_length,
            'prediction_length': args.prediction_length,
        }

        count = all_stat[-1]
        for i, metric in enumerate(metric_list):
            val = all_stat[i] / count
            item[metric.name] = float(val.cpu().numpy())
        logging.info(item)

    # Export routing stats
    if export_routing and rank == 0:
        routing_output = _format_routing_stats(routing_stats, model.model)
        out_path = getattr(args, 'routing_stats_path', None) or os.path.join(
            os.path.dirname(args.model) if os.path.isdir(args.model) else '.', 'routing_stats.json'
        )
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(routing_output, f, indent=2, default=str)
        logging.info(f'Routing stats saved to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TimeMoE Evaluate')
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='Maple728/TimeMoE-50M',
        help='Model path'
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        help='Benchmark data path'
    )

    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=32,
        help='Batch size of evaluation'
    )
    parser.add_argument(
        '--context_length', '-c',
        type=int,
        default=None,
        help='Context length'
    )
    parser.add_argument(
        '--export_routing_stats',
        action='store_true',
        default=False,
        help='Export typed routing statistics to JSON after evaluation'
    )
    parser.add_argument(
        '--routing_stats_path',
        type=str,
        default=None,
        help='Output path for routing stats JSON (default: routing_stats.json next to model)'
    )
    parser.add_argument(
        '--prediction_length', '-p',
        type=int,
        default=96,
        help='Prediction length'
    )
    args = parser.parse_args()
    if args.context_length is None:
        if args.prediction_length == 96:
            args.context_length = 512
        elif args.prediction_length == 192:
            args.context_length = 1024
        elif args.prediction_length == 336:
            args.context_length = 2048
        elif args.prediction_length == 720:
            args.context_length = 3072
        else:
            args.context_length = args.prediction_length * 4
    evaluate(args)
