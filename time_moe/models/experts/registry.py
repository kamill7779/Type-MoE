from typing import Callable, Dict, Any

from .mlp_temporal_block_expert import MLPTemporalBlockExpert


EXPERT_REGISTRY: Dict[str, Callable[..., object]] = {}


def register_expert(name: str):
    def _wrap(cls):
        EXPERT_REGISTRY[name] = cls
        return cls
    return _wrap


@register_expert("mlp_temporal_block")
class RegisteredMLPTemporalBlockExpert(MLPTemporalBlockExpert):
    pass


def _normalize_spec(spec: Any) -> Dict[str, Any]:
    if isinstance(spec, str):
        return {"name": spec, "params": {}}
    if isinstance(spec, dict):
        name = spec.get("name")
        if not name:
            raise ValueError("Expert spec dict must include 'name'")
        return {
            "name": name,
            "params": spec.get("params", {}),
            "type": spec.get("type"),
            "interface": spec.get("interface"),
            "zero_init_output": spec.get("zero_init_output", False),
        }
    raise ValueError(f"Unsupported expert spec type: {type(spec)}")


def build_expert(
        spec: Any,
        *,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        output_norm: bool = True,
):
    normalized = _normalize_spec(spec)
    name = normalized["name"]
    if name not in EXPERT_REGISTRY:
        raise ValueError(f"Unknown expert '{name}'. Registered: {sorted(EXPERT_REGISTRY.keys())}")

    cls = EXPERT_REGISTRY[name]
    params = dict(normalized.get("params") or {})
    params.setdefault("hidden_size", hidden_size)
    params.setdefault("intermediate_size", intermediate_size)
    params.setdefault("hidden_act", hidden_act)
    params.setdefault("output_norm", output_norm)
    expert = cls(**params)

    if normalized.get("type") is not None:
        expert.expert_type = normalized["type"]
    if normalized.get("interface") is not None:
        expert.interface_kind = normalized["interface"]
    if normalized.get("zero_init_output") and hasattr(expert, "zero_init_output"):
        expert.zero_init_output()
    return expert
