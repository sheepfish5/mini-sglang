from __future__ import annotations

from pathlib import Path

import safetensors
import torch
import torch.nn.functional as F

from minisgl.distributed.info import set_tp_info, try_get_tp_info
from minisgl.layers.linear import LinearQKVMerged
from minisgl.models.weight import _shard_tensor


MODEL_FILE = Path("/mnt/c/Users/sheepfish5/Downloads/model-00001-of-00050.safetensors")

Q_KEY = "language_model.model.layers.0.self_attn.q_proj.weight"
K_KEY = "language_model.model.layers.0.self_attn.k_proj.weight"
V_KEY = "language_model.model.layers.0.self_attn.v_proj.weight"

HIDDEN_PATH = Path("tmp/l0_before_attn_hidden_states.pt")
MINI_QKV_PATH = Path("tmp/l0_attn_after_qkv_proj.pt")
SGLANG_QKV_PATH = Path("tmp/sglang_l0_attn_after_qkv_proj.pt")
SGLANG_WEIGHT_PATH = Path("tmp/sglang_l0_qkv_proj_weight.pt")

TP_RANK = 0
TP_SIZE = 4
HIDDEN_SIZE = 5120
HEAD_DIM = 128
NUM_QO_HEADS = 40
NUM_KV_HEADS = 8


def report_diff(name: str, lhs: torch.Tensor, rhs: torch.Tensor):
    diff = (lhs.float() - rhs.float()).abs()
    print(
        f"{name}: shape={tuple(lhs.shape)}, "
        f"max_abs_diff={diff.max().item():.6g}, "
        f"mean_abs_diff={diff.mean().item():.6g}"
    )


def _reshape_heads(weight: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    return weight.view(num_heads, head_dim, weight.shape[-1])


def permute_head_rows(weight: torch.Tensor, num_heads: int, head_dim: int, mode: str) -> torch.Tensor:
    heads = _reshape_heads(weight, num_heads, head_dim)

    if mode == "half_to_interleave":
        first, second = heads.chunk(2, dim=1)
        out = torch.stack((first, second), dim=2).reshape_as(heads)
    elif mode == "interleave_to_half":
        even = heads[:, ::2, :]
        odd = heads[:, 1::2, :]
        out = torch.cat((even, odd), dim=1)
    elif mode == "swap_halves":
        first, second = heads.chunk(2, dim=1)
        out = torch.cat((second, first), dim=1)
    elif mode == "pair_swap":
        out = heads.view(num_heads, head_dim // 2, 2, heads.shape[-1]).flip(2).reshape_as(heads)
    else:
        raise ValueError(f"Unknown permutation mode: {mode}")

    return out.reshape_as(weight)


def test_candidate_permutations(
    prefix: str,
    weight: torch.Tensor,
    sglang_weight: torch.Tensor,
    hidden_states: torch.Tensor,
    sglang_out: torch.Tensor,
    num_heads: int,
    head_dim: int,
):
    print(f"{prefix}_candidate_permutations:")
    for mode in ("half_to_interleave", "interleave_to_half", "swap_halves", "pair_swap"):
        perm_weight = permute_head_rows(weight, num_heads, head_dim, mode)
        out = F.linear(hidden_states, perm_weight.to(device=hidden_states.device, dtype=hidden_states.dtype))
        weight_diff = (perm_weight.float() - sglang_weight.float()).abs()
        out_diff = (out.float() - sglang_out.float()).abs()
        print(
            f"{prefix}_{mode}: "
            f"weight_max={weight_diff.max().item():.6g}, "
            f"weight_mean={weight_diff.mean().item():.6g}, "
            f"out_max={out_diff.max().item():.6g}, "
            f"out_mean={out_diff.mean().item():.6g}"
        )


def main():
    tp_info = try_get_tp_info()
    if tp_info is None:
        set_tp_info(TP_RANK, TP_SIZE)
    else:
        print(f"existing_tp_info=={tp_info}")

    with safetensors.safe_open(MODEL_FILE, framework="pt", device="cpu") as f:
        q_full = f.get_tensor(Q_KEY)
        k_full = f.get_tensor(K_KEY)
        v_full = f.get_tensor(V_KEY)

    print(f"q_full.shape=={q_full.shape}")
    print(f"k_full.shape=={k_full.shape}")
    print(f"v_full.shape=={v_full.shape}")

    q_shard = _shard_tensor(Q_KEY, q_full, TP_RANK, TP_SIZE, is_llama4=True)
    k_shard = _shard_tensor(K_KEY, k_full, TP_RANK, TP_SIZE, is_llama4=True)
    v_shard = _shard_tensor(V_KEY, v_full, TP_RANK, TP_SIZE, is_llama4=True)
    merged_shard = torch.cat([q_shard, k_shard, v_shard], dim=0)

    print(f"q_shard.shape=={q_shard.shape}")
    print(f"k_shard.shape=={k_shard.shape}")
    print(f"v_shard.shape=={v_shard.shape}")
    print(f"merged_shard.shape=={merged_shard.shape}")

    qkv_proj = LinearQKVMerged(
        hidden_size=HIDDEN_SIZE,
        head_dim=HEAD_DIM,
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        has_bias=False,
    )
    qkv_proj.weight = merged_shard

    hidden_states = torch.load(HIDDEN_PATH)
    mini_qkv_saved = torch.load(MINI_QKV_PATH)
    sglang_qkv_saved = torch.load(SGLANG_QKV_PATH)
    sglang_weight_saved = torch.load(
        SGLANG_WEIGHT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    print(f"hidden_states.shape=={hidden_states.shape}, dtype=={hidden_states.dtype}")
    print(f"mini_qkv_saved.shape=={mini_qkv_saved.shape}, dtype=={mini_qkv_saved.dtype}")
    print(
        f"sglang_qkv_saved.shape=={sglang_qkv_saved.shape}, dtype=={sglang_qkv_saved.dtype}"
    )
    print(
        f"sglang_weight_saved.shape=={sglang_weight_saved.shape}, "
        f"dtype=={sglang_weight_saved.dtype}"
    )

    device = hidden_states.device
    qkv_proj.weight = qkv_proj.weight.to(device=device, dtype=hidden_states.dtype)

    qkv_out = qkv_proj.forward(hidden_states)
    print(f"qkv_out.shape=={qkv_out.shape}, dtype=={qkv_out.dtype}")

    report_diff("constructed_vs_mini_qkv", qkv_out, mini_qkv_saved)
    report_diff("constructed_vs_sglang_qkv", qkv_out, sglang_qkv_saved)
    report_diff("mini_vs_sglang_qkv", mini_qkv_saved, sglang_qkv_saved)
    report_diff(
        "constructed_weight_vs_sglang_weight",
        merged_shard.float(),
        sglang_weight_saved.float(),
    )

    q_dim = mini_qkv_saved.shape[-1] - 2 * (k_shard.shape[0])
    kv_dim = k_shard.shape[0]
    q_out, k_out, v_out = qkv_out.split([q_dim, kv_dim, kv_dim], dim=-1)
    mini_q, mini_k, mini_v = mini_qkv_saved.split([q_dim, kv_dim, kv_dim], dim=-1)
    sglang_q, sglang_k, sglang_v = sglang_qkv_saved.split([q_dim, kv_dim, kv_dim], dim=-1)

    report_diff("constructed_q_vs_mini_q", q_out, mini_q)
    report_diff("constructed_k_vs_mini_k", k_out, mini_k)
    report_diff("constructed_v_vs_mini_v", v_out, mini_v)
    report_diff("constructed_q_vs_sglang_q", q_out, sglang_q)
    report_diff("constructed_k_vs_sglang_k", k_out, sglang_k)
    report_diff("constructed_v_vs_sglang_v", v_out, sglang_v)

    w_q, w_k, w_v = merged_shard.split([q_dim, kv_dim, kv_dim], dim=0)
    s_w_q, s_w_k, s_w_v = sglang_weight_saved.split([q_dim, kv_dim, kv_dim], dim=0)
    report_diff("constructed_wq_vs_sglang_wq", w_q.float(), s_w_q.float())
    report_diff("constructed_wk_vs_sglang_wk", w_k.float(), s_w_k.float())
    report_diff("constructed_wv_vs_sglang_wv", w_v.float(), s_w_v.float())

    q_heads = q_out.view(-1, q_dim // HEAD_DIM, HEAD_DIM)
    mini_q_heads = mini_q.view(-1, q_dim // HEAD_DIM, HEAD_DIM)
    sglang_q_heads = sglang_q.view(-1, q_dim // HEAD_DIM, HEAD_DIM)
    print(
        f"constructed_vs_mini_q_head_diff=="
        f"{(q_heads - mini_q_heads).abs().amax(dim=(0, 2))}"
    )
    print(
        f"constructed_vs_sglang_q_head_diff=="
        f"{(q_heads - sglang_q_heads).abs().amax(dim=(0, 2))}"
    )

    w_q_heads = w_q.view(q_dim // HEAD_DIM, HEAD_DIM, HIDDEN_SIZE)
    s_w_q_heads = s_w_q.view(q_dim // HEAD_DIM, HEAD_DIM, HIDDEN_SIZE)
    w_k_heads = w_k.view(kv_dim // HEAD_DIM, HEAD_DIM, HIDDEN_SIZE)
    s_w_k_heads = s_w_k.view(kv_dim // HEAD_DIM, HEAD_DIM, HIDDEN_SIZE)
    print(
        f"constructed_wq_vs_sglang_wq_head_diff=="
        f"{(w_q_heads - s_w_q_heads).abs().amax(dim=(1, 2))}"
    )
    print(
        f"constructed_wk_vs_sglang_wk_head_diff=="
        f"{(w_k_heads - s_w_k_heads).abs().amax(dim=(1, 2))}"
    )

    test_candidate_permutations(
        prefix="q",
        weight=w_q,
        sglang_weight=s_w_q,
        hidden_states=hidden_states,
        sglang_out=sglang_q,
        num_heads=q_dim // HEAD_DIM,
        head_dim=HEAD_DIM,
    )
    test_candidate_permutations(
        prefix="k",
        weight=w_k,
        sglang_weight=s_w_k,
        hidden_states=hidden_states,
        sglang_out=sglang_k,
        num_heads=kv_dim // HEAD_DIM,
        head_dim=HEAD_DIM,
    )

    full_merged = torch.cat([q_full, k_full, v_full], dim=0)
    naive_chunk = full_merged.chunk(TP_SIZE, dim=0)[TP_RANK].clone()
    print(f"naive_chunk.shape=={naive_chunk.shape}")
    report_diff("mini_merge_vs_naive_chunk", merged_shard, naive_chunk)

    topk = min(20, qkv_out.numel())
    flat_diff = (qkv_out.float() - sglang_qkv_saved.float()).abs().flatten()
    top_vals, top_idx = torch.topk(flat_diff, k=topk)
    out_dim = qkv_out.shape[-1]
    print("top_constructed_vs_sglang_qkv_diff:")
    for rank, (val, idx) in enumerate(zip(top_vals.tolist(), top_idx.tolist()), start=1):
        token_idx = idx // out_dim
        hidden_idx = idx % out_dim
        print(
            f"{rank}: token={token_idx}, out={hidden_idx}, abs_diff={val:.6g}, "
            f"constructed={qkv_out[token_idx, hidden_idx].item()}, "
            f"sglang={sglang_qkv_saved[token_idx, hidden_idx].item()}"
        )

    flat_w_diff = (merged_shard.float() - sglang_weight_saved.float()).abs().flatten()
    top_w_vals, top_w_idx = torch.topk(flat_w_diff, k=min(20, flat_w_diff.numel()))
    print("top_constructed_vs_sglang_weight_diff:")
    for rank, (val, idx) in enumerate(zip(top_w_vals.tolist(), top_w_idx.tolist()), start=1):
        out_idx = idx // HIDDEN_SIZE
        in_idx = idx % HIDDEN_SIZE
        print(
            f"{rank}: out={out_idx}, in={in_idx}, abs_diff={val:.6g}, "
            f"constructed={merged_shard[out_idx, in_idx].item()}, "
            f"sglang={sglang_weight_saved[out_idx, in_idx].item()}"
        )


if __name__ == "__main__":
    main()
