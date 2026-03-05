from typing import List
from transformers import PretrainedConfig


class TimeMoeConfig(PretrainedConfig):
    model_type = "time_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
            self,
            input_size: int = 1,
            hidden_size: int = 4096,
            intermediate_size: int = 22016,
            horizon_lengths: List[int] = 1,
            num_hidden_layers: int = 32,
            num_attention_heads: int = 32,
            num_key_value_heads: int = None,
            hidden_act: str = "silu",
            num_experts_per_tok: int = 2,
            num_experts: int = 1,
            max_position_embeddings: int = 32768,
            initializer_range: float = 0.02,
            rms_norm_eps: float = 1e-6,
            use_cache: bool = True,
            use_dense: bool = False,
            rope_theta: int = 10000,
            attention_dropout: float = 0.0,
            apply_aux_loss: bool = True,
            router_aux_loss_factor: float = 0.02,
            router_mode: str = "standard",
            expert_types: List[str] = None,
            expert_type_map: List[int] = None,
            norm_topk_prob: bool = False,
            jitter_noise: float = 0.0,
            type_diversity_factor: float = 0.0,
            seq_expert_mode: str = "full_seq",
            seq_expert_window: int = 64,
            expert_output_norm: bool = True,
            custom_expert_specs: List[dict] = None,
            freeze_strategy: str = "none",
            tie_word_embeddings: bool = False,
            **kwargs,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        if isinstance(horizon_lengths, int):
            horizon_lengths = [horizon_lengths]
        self.horizon_lengths = horizon_lengths  # Predict horizon length for each prediction.
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.use_dense = use_dense
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.apply_aux_loss = apply_aux_loss
        self.router_aux_loss_factor = router_aux_loss_factor
        self.router_mode = router_mode
        self.expert_types = expert_types if expert_types is not None else []
        self.expert_type_map = expert_type_map if expert_type_map is not None else []
        self.norm_topk_prob = norm_topk_prob
        self.jitter_noise = jitter_noise
        self.type_diversity_factor = type_diversity_factor
        self.seq_expert_mode = seq_expert_mode
        self.seq_expert_window = seq_expert_window
        self.expert_output_norm = expert_output_norm
        self.custom_expert_specs = custom_expert_specs if custom_expert_specs is not None else []
        self.freeze_strategy = freeze_strategy

        assert self.use_dense ^ self.apply_aux_loss, 'Both use_dense and apply_aux_loss cannot be set to True or False at the same time.'
        if self.router_mode not in ["standard", "typed_topk"]:
            raise ValueError(f"Unsupported router_mode: {self.router_mode}")
        if len(self.expert_type_map) > 0 and len(self.expert_type_map) != self.num_experts:
            raise ValueError(
                f"expert_type_map size mismatch: expected {self.num_experts}, got {len(self.expert_type_map)}"
            )
        if self.seq_expert_mode not in ["full_seq", "local_window"]:
            raise ValueError(f"Unsupported seq_expert_mode: {self.seq_expert_mode}")
        if self.freeze_strategy not in ["none", "phased", "gate_only"]:
            raise ValueError(f"Unsupported freeze_strategy: {self.freeze_strategy}")

        kwargs.pop('tie_word_embeddings', None)
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
