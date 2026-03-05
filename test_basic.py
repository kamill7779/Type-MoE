#!/usr/bin/env python
"""Basic test script to verify project structure and model loading."""
import sys
import os
import torch
import traceback

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all project modules can be imported."""
    print("=" * 60)
    print("[Test 1] Testing project imports...")
    try:
        from time_moe.models.configuration_time_moe import TimeMoeConfig
        print("  [OK] TimeMoeConfig imported")
    except Exception as e:
        print(f"  [FAIL] TimeMoeConfig: {e}")
        traceback.print_exc()
        return False

    try:
        from time_moe.models.modeling_time_moe import TimeMoeForPrediction
        print("  [OK] TimeMoeForPrediction imported")
    except Exception as e:
        print(f"  [FAIL] TimeMoeForPrediction: {e}")
        traceback.print_exc()
        return False

    try:
        from time_moe.models.typed_router_utils import (
            RouterInfo, typed_preselect, _resolve_actual_k,
            _collect_routing_stats, load_balancing_loss_func, type_diversity_loss_func
        )
        print("  [OK] typed_router_utils imported")
    except Exception as e:
        print(f"  [FAIL] typed_router_utils: {e}")
        traceback.print_exc()
        return False

    try:
        from time_moe.models.experts.registry import build_expert
        print("  [OK] experts.registry imported")
    except Exception as e:
        print(f"  [FAIL] experts.registry: {e}")
        traceback.print_exc()
        return False

    try:
        from time_moe.models.experts.base import BaseTokenExpert
        from time_moe.models.experts.anomaly_token_expert import AnomalyTokenExpert
        from time_moe.models.experts.autoformer_token_expert import AutoformerTrendExpert, AutoformerCycleExpert
        from time_moe.models.experts.fedformer_token_expert import FedFormerCycleExpert
        from time_moe.models.experts.nbeats_token_expert import NBeatsTokenExpert
        from time_moe.models.experts.mlp_temporal_block_expert import MLPTemporalBlockExpert
        print("  [OK] All expert modules imported")
    except Exception as e:
        print(f"  [FAIL] Expert modules: {e}")
        traceback.print_exc()
        return False

    print("  All imports passed!")
    return True


def test_load_original_model():
    """Test loading the original TimeMoE-50M model."""
    print("=" * 60)
    print("[Test 2] Testing original TimeMoE-50M model loading...")
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            "./TimeMoE-50M",
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        print(f"  [OK] Model loaded: {type(model).__name__}")
        print(f"  [OK] Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model
    except Exception as e:
        print(f"  [FAIL] Model loading: {e}")
        traceback.print_exc()
        return None


def test_forward_pass(model):
    """Test a simple forward pass."""
    print("=" * 60)
    print("[Test 3] Testing forward pass...")
    try:
        # Create dummy input: batch of 2, sequence length 64
        input_ids = torch.randn(2, 64)
        with torch.no_grad():
            output = model(input_ids=input_ids)
        print(f"  [OK] Forward pass successful")
        print(f"  [OK] Output logits shape: {output.logits.shape}")
        if hasattr(output, 'router_logits') and output.router_logits:
            print(f"  [OK] Router logits: {len(output.router_logits)} layers")
        return True
    except Exception as e:
        print(f"  [FAIL] Forward pass: {e}")
        traceback.print_exc()
        return False


def test_typed_config():
    """Test creating a config with typed routing features."""
    print("=" * 60)
    print("[Test 4] Testing typed routing config...")
    try:
        from time_moe.models.configuration_time_moe import TimeMoeConfig
        config = TimeMoeConfig(
            hidden_size=384,
            intermediate_size=1536,
            num_hidden_layers=2,
            num_attention_heads=12,
            num_experts=8,
            num_experts_per_tok=2,
            router_mode="typed_topk",
            expert_types=["trend", "cycle", "anomaly"],
            expert_type_map=[0, 0, 0, 1, 1, 1, 2, 2],
            norm_topk_prob=False,
            jitter_noise=0.02,
            type_diversity_factor=0.01,
            custom_expert_specs=[
                {"name": "mlp_temporal_block", "type": "trend", "interface": "flat"},
                {"name": "nbeats_trend", "type": "trend", "interface": "flat",
                 "zero_init_output": True, "params": {"num_layers": 4, "theta_dim": 16}},
                {"name": "autoformer_trend", "type": "trend", "interface": "seq",
                 "zero_init_output": True, "params": {"kernel_size": 25}},
                {"name": "mlp_temporal_block", "type": "cycle", "interface": "flat"},
                {"name": "autoformer_cycle", "type": "cycle", "interface": "seq",
                 "zero_init_output": True, "params": {"kernel_size": 25, "top_k_freq": 3}},
                {"name": "fedformer_cycle", "type": "cycle", "interface": "seq",
                 "zero_init_output": True, "params": {"modes": 32}},
                {"name": "mlp_temporal_block", "type": "anomaly", "interface": "flat"},
                {"name": "anomaly_attn", "type": "anomaly", "interface": "seq",
                 "zero_init_output": True, "params": {"num_heads": 8}},
            ],
            horizon_lengths=[1, 8, 32, 64],
        )
        print(f"  [OK] Config created: router_mode={config.router_mode}")
        print(f"  [OK] Expert types: {config.expert_types}")
        print(f"  [OK] Expert type map: {config.expert_type_map}")
        print(f"  [OK] Custom expert specs: {len(config.custom_expert_specs)} experts")
        return config
    except Exception as e:
        print(f"  [FAIL] Config creation: {e}")
        traceback.print_exc()
        return None


def test_typed_model(config):
    """Test instantiating a model with typed routing from scratch."""
    print("=" * 60)
    print("[Test 5] Testing typed routing model instantiation...")
    try:
        from time_moe.models.modeling_time_moe import TimeMoeForPrediction
        model = TimeMoeForPrediction(config)
        print(f"  [OK] Model instantiated: {type(model).__name__}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  [OK] Model parameters: {total_params:,}")

        # Test forward pass
        input_ids = torch.randn(2, 64)
        with torch.no_grad():
            output = model(input_ids=input_ids)
        print(f"  [OK] Forward pass successful")
        print(f"  [OK] Output logits shape: {output.logits.shape}")
        return True
    except Exception as e:
        print(f"  [FAIL] Typed model: {e}")
        traceback.print_exc()
        return False


def test_expert_build():
    """Test building individual experts via registry."""
    print("=" * 60)
    print("[Test 6] Testing expert registry build...")
    try:
        from time_moe.models.experts.registry import build_expert
        hidden_size = 384
        intermediate_size = 1536

        specs = [
            {"name": "mlp_temporal_block", "type": "trend", "interface": "flat"},
            {"name": "nbeats_trend", "type": "trend", "interface": "flat",
             "zero_init_output": True, "params": {"num_layers": 4, "theta_dim": 16}},
            {"name": "autoformer_trend", "type": "trend", "interface": "seq",
             "zero_init_output": True, "params": {"kernel_size": 25}},
            {"name": "autoformer_cycle", "type": "cycle", "interface": "seq",
             "zero_init_output": True, "params": {"kernel_size": 25, "top_k_freq": 3}},
            {"name": "fedformer_cycle", "type": "cycle", "interface": "seq",
             "zero_init_output": True, "params": {"modes": 32}},
            {"name": "anomaly_attn", "type": "anomaly", "interface": "seq",
             "zero_init_output": True, "params": {"num_heads": 8}},
        ]

        for spec in specs:
            expert = build_expert(
                spec=spec,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                hidden_act="silu",
                output_norm=True,
            )
            # Quick forward test: seq experts need 3D [B, L, H], flat experts need 2D [N, H]
            iface = spec.get("interface", "flat")
            if iface == "seq":
                x = torch.randn(2, 16, hidden_size)  # [B, L, H]
            else:
                x = torch.randn(4, hidden_size)  # [N, H]
            with torch.no_grad():
                out = expert(x)
            print(f"  [OK] {spec['name']} ({iface}) built and forward tested, output shape: {out.shape}")

        return True
    except Exception as e:
        print(f"  [FAIL] Expert build: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    results = {}

    results["imports"] = test_imports()
    results["expert_build"] = test_expert_build()
    results["original_model"] = test_load_original_model()
    if results["original_model"]:
        results["forward_pass"] = test_forward_pass(results["original_model"])
    config = test_typed_config()
    if config:
        results["typed_model"] = test_typed_model(config)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    all_passed = all(v for v in results.values() if isinstance(v, bool))
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_passed else 1)
