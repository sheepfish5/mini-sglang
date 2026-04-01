import torch

from minisgl.layers.rotary import get_rope

num_qo_heads = 10
num_kv_heads = 2
head_dim = 128
qo_attn_dim = num_qo_heads * head_dim
kv_attn_dim = num_kv_heads * head_dim

# --- mini-sglang qkv ---
mini_qkv: torch.Tensor = torch.load("tmp/l0_attn_after_qkv_proj.pt")
print(f"mini_qkv.shape=={mini_qkv.shape}")

mini_q, mini_k, mini_v = mini_qkv.split([qo_attn_dim, kv_attn_dim, kv_attn_dim], dim=-1)
print(f"min_q.shape=={mini_q.shape}")
print(f"min_k.shape=={mini_k.shape}")
print(f"min_v.shape=={mini_v.shape}")

mini_q_sum = torch.sum(mini_q, dim=-1)
mini_k_sum = torch.sum(mini_k, dim=-1)
print(f"mini_q_sum=={mini_q_sum}")
print(f"mini_k_sum=={mini_k_sum}")

# --- sglang qkv

sglang_qkv: torch.Tensor = torch.load("tmp/sglang_l0_attn_after_qkv_proj.pt")
print(f"sglang_qkv.shape=={sglang_qkv.shape}")

sglang_q, sglang_k, sglang_v = sglang_qkv.split([qo_attn_dim, kv_attn_dim, kv_attn_dim], dim=-1)
print(f"min_q.shape=={sglang_q.shape}")
print(f"min_k.shape=={sglang_k.shape}")
print(f"min_v.shape=={sglang_v.shape}")

sglang_q_sum = torch.sum(sglang_q, dim=-1)
sglang_k_sum = torch.sum(sglang_k, dim=-1)
print(f"sglang_q_sum=={sglang_q_sum}")
print(f"sglang_k_sum=={sglang_k_sum}")


# --- mini-sglang rope

from minisgl.models.config import RotaryConfig

# max_position_embeddings = 10485760
max_position_embeddings = 1024
rope_theta = 500000.0
rope_scaling = {
    "factor": 16.0,
    "high_freq_factor": 1.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3",
}

rotary_config = RotaryConfig(
    head_dim=head_dim,
    rotary_dim=head_dim,
    max_position=max_position_embeddings,
    base=rope_theta,
    scaling=rope_scaling,
)

mini_rotary = get_rope(
    head_dim=head_dim,
    rotary_dim=rotary_config.rotary_dim,
    max_position=rotary_config.max_position,
    base=rotary_config.base,
    rope_scaling=tuple(rotary_config.scaling.items()) if rotary_config.scaling else None,
)
mini_rotary._cos_sin_cache = mini_rotary._cos_sin_cache.to(device=mini_q.device)

positions = torch.arange(1, 14, device=mini_qkv.device)
print(f"positions=={positions}")

import os
os.environ["MINISGL_DEBUG_ROPE_REF"] = "1"

mini_rotary.forward(positions, mini_q, mini_k)

mini_q_sum = torch.sum(mini_q, dim=-1)
mini_k_sum = torch.sum(mini_k, dim=-1)
print(f"mini_q_sum=={mini_q_sum}")
print(f"mini_k_sum=={mini_k_sum}")

mini_q_ar: torch.Tensor = torch.load("tmp/l0_after_rope_q.pt")
mini_k_ar: torch.Tensor = torch.load("tmp/l0_after_rope_k.pt")
print(f"mini_q_ar.shape=={mini_q_ar.shape}")
print(f"mini_k_ar.shape=={mini_k_ar.shape}")

mini_q_ar_sum = torch.sum(mini_q_ar, dim=-1)
mini_k_ar_sum = torch.sum(mini_k_ar, dim=-1)
print(f"mini_q_ar_sum=={mini_q_ar_sum}")
print(f"mini_k_ar_sum=={mini_k_ar_sum}")

# --- sglang after rope q k

sglang_q_ar: torch.Tensor = torch.load("tmp/sglang_l0_after_rope_q.pt")
sglang_k_ar: torch.Tensor = torch.load("tmp/sglang_l0_after_rope_k.pt")
print(f"sglang_q_ar.shape=={sglang_q_ar.shape}")
print(f"sglang_k_ar.shape=={sglang_k_ar.shape}")

sglang_q_ar_sum = torch.sum(sglang_q_ar, dim=-1)
sglang_k_ar_sum = torch.sum(sglang_k_ar, dim=-1)
print(f"sglang_q_ar_sum=={sglang_q_ar_sum}")
print(f"sglang_k_ar_sum=={sglang_k_ar_sum}")