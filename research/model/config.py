"""Ornith 35B MoE configuration from HF model card."""

n_layers = 28
n_experts = 256
n_active_experts = 8
has_shared_expert = True
d_model = 2560
n_heads = 20
n_kv_heads = 4
head_dim = 128
hidden_dim = 6912
router_hidden_dim = 1536

# MoE MLP intermediate size per expert
expert_hidden_dim = 2048

# Attention
attention_type = "hybrid"  # full attention + Gated DeltaNet
rope_theta = 1000000.0
rope_scaling = "yarn"
max_seq_len = 262144

# Quantization (plan)
expert_quant = "int4"     # routed experts
resident_quant = "int8"   # non-routed weights
router_quant = "fp16"

# Memory estimates (bytes)
def param_size(n_params, bits):
    return n_params * bits // 8

# Per-layer expert (all 256)
expert_params = 2 * hidden_dim * expert_hidden_dim  # gate+up, down = ~13M params
expert_size_int4 = param_size(expert_params, 4)  # ~6.5 MB per expert
expert_size_int8 = param_size(expert_params, 8)  # ~13 MB

# Non-routed per layer
attn_qkv_proj = d_model * head_dim * (n_heads + 2 * n_kv_heads)
attn_o_proj = d_model * d_model
shared_expert_params = 2 * hidden_dim * hidden_dim
norms_per_layer = 2 * d_model
non_routed_per_layer = attn_qkv_proj + attn_o_proj + shared_expert_params + norms_per_layer

# Total non-routed resident size (int8)
non_routed_total = (non_routed_per_layer * n_layers
                    + d_model * n_heads  # embeddings
                    + d_model * n_heads)  # lm_head
non_routed_total_bytes = non_routed_total  # int8 = 1 byte/param

# Routed experts total (int4)
routed_total_bytes = expert_size_int4 * n_experts * n_layers
