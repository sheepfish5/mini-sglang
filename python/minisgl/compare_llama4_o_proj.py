from __future__ import annotations

from pathlib import Path

import safetensors
import torch
import torch.nn.functional as F

from minisgl.models.weight import _shard_tensor


MINI_INPUT_PATH = Path("tmp/l0_attn_after_attention.pt")
SGLANG_INPUT_PATH = Path("tmp/sglang_l0_attn_after_attention.pt")
MINI_OUTPUT_PATH = Path("tmp/l0_after_attn_hidden_states.pt")
SGLANG_OUTPUT_PATH = Path("tmp/sglang_l0_after_attn_hidden_states.pt")
MINI_WEIGHT_PATH = Path("tmp/l0_o_proj_weight.pt")
SGLANG_WEIGHT_PATH = Path("tmp/sglang_l0_o_proj_weight.pt")
MODEL_FILE = Path("/mnt/c/Users/sheepfish5/Downloads/model-00001-of-00050.safetensors")
O_PROJ_KEY = "language_model.model.layers.0.self_attn.o_proj.weight"

NUM_Q_HEADS = 10
HEAD_DIM = 128
TP_RANK = 0
TP_SIZE = 4


def permute_head_cols(weight: torch.Tensor, num_heads: int, head_dim: int, mode: str) -> torch.Tensor:
    heads = weight.view(weight.shape[0], num_heads, head_dim)

    if mode == "half_to_interleave":
        first, second = heads.chunk(2, dim=2)
        out = torch.stack((first, second), dim=3).reshape_as(heads)
    elif mode == "interleave_to_half":
        even = heads[:, :, ::2]
        odd = heads[:, :, 1::2]
        out = torch.cat((even, odd), dim=2)
    elif mode == "swap_halves":
        first, second = heads.chunk(2, dim=2)
        out = torch.cat((second, first), dim=2)
    elif mode == "pair_swap":
        out = heads.view(weight.shape[0], num_heads, head_dim // 2, 2).flip(3).reshape_as(heads)
    else:
        raise ValueError(f"Unknown permutation mode: {mode}")

    return out.reshape_as(weight)


def maybe_strip_leading_bos(mini: torch.Tensor, sglang: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if mini.ndim > 0 and sglang.ndim > 0 and mini.shape[0] == sglang.shape[0] + 1:
        return mini[1:], sglang, True
    return mini, sglang, False


def print_alignment_note(name: str, stripped: bool) -> None:
    if stripped:
        print(f"{name}: stripped leading BOS row from mini before comparison")


def report_diff(name: str, lhs: torch.Tensor, rhs: torch.Tensor):
    diff = (lhs.float() - rhs.float()).abs()
    print(
        f"{name}: shape={tuple(lhs.shape)}, "
        f"max_abs_diff={diff.max().item():.6g}, "
        f"mean_abs_diff={diff.mean().item():.6g}"
    )


def report_aligned_diff(name: str, mini: torch.Tensor, sglang: torch.Tensor):
    mini_aligned, sglang_aligned, stripped = maybe_strip_leading_bos(mini, sglang)
    print_alignment_note(name, stripped)
    report_diff(name, mini_aligned, sglang_aligned)


def report_head_diff(name: str, lhs: torch.Tensor, rhs: torch.Tensor):
    lhs_h = lhs.float().view(-1, NUM_Q_HEADS, HEAD_DIM)
    rhs_h = rhs.float().view(-1, NUM_Q_HEADS, HEAD_DIM)
    diff = (lhs_h - rhs_h).abs().amax(dim=(0, 2))
    print(f"{name}_head_diff=={diff}")


def report_top_diff(name: str, lhs: torch.Tensor, rhs: torch.Tensor, limit: int = 20):
    diff = (lhs.float() - rhs.float()).abs().flatten()
    k = min(limit, diff.numel())
    vals, idxs = torch.topk(diff, k=k)
    last_dim = lhs.shape[-1]
    print(f"top_{name}_diff:")
    for rank, (val, idx) in enumerate(zip(vals.tolist(), idxs.tolist()), start=1):
        row = idx // last_dim
        col = idx % last_dim
        print(
            f"{rank}: row={row}, col={col}, abs_diff={val:.6g}, "
            f"lhs={lhs.flatten()[idx].item()}, rhs={rhs.flatten()[idx].item()}"
        )


def main():
    mini_in = torch.load(MINI_INPUT_PATH, map_location="cpu")
    sglang_in = torch.load(SGLANG_INPUT_PATH, map_location="cpu")
    mini_out_saved = torch.load(MINI_OUTPUT_PATH, map_location="cpu")
    sglang_out_saved = torch.load(SGLANG_OUTPUT_PATH, map_location="cpu")
    mini_weight = torch.load(MINI_WEIGHT_PATH, map_location="cpu")
    sglang_weight = torch.load(SGLANG_WEIGHT_PATH, map_location="cpu")
    with safetensors.safe_open(MODEL_FILE, framework="pt", device="cpu") as f:
        full_weight = f.get_tensor(O_PROJ_KEY)
    sharded_weight = _shard_tensor(O_PROJ_KEY, full_weight, TP_RANK, TP_SIZE)


    print(f"mini_in.shape=={mini_in.shape}, dtype=={mini_in.dtype}")
    print(f"sglang_in.shape=={sglang_in.shape}, dtype=={sglang_in.dtype}")
    print(f"mini_out_saved.shape=={mini_out_saved.shape}, dtype=={mini_out_saved.dtype}")
    print(f"sglang_out_saved.shape=={sglang_out_saved.shape}, dtype=={sglang_out_saved.dtype}")
    print(f"mini_weight.shape=={mini_weight.shape}, dtype=={mini_weight.dtype}")
    print(f"sglang_weight.shape=={sglang_weight.shape}, dtype=={sglang_weight.dtype}")
    print(f"full_weight.shape=={full_weight.shape}, dtype=={full_weight.dtype}")
    print(f"sharded_weight.shape=={sharded_weight.shape}, dtype=={sharded_weight.dtype}")

    report_aligned_diff("attn_input", mini_in, sglang_in)
    report_diff("o_proj_weight", mini_weight, sglang_weight)
    report_diff("hf_sharded_weight_vs_mini_weight", sharded_weight, mini_weight)
    report_diff("hf_sharded_weight_vs_sglang_weight", sharded_weight, sglang_weight)


    # LinearOProj / RowParallelLinear are row-parallel:
    # saved weight is a local shard, and the real forward includes an all-reduce.
    # So F.linear(input, weight) below is only the local partial output, not the final layer output.
    mini_local = F.linear(mini_in, mini_weight.to(dtype=mini_in.dtype))
    sglang_local = F.linear(sglang_in, sglang_weight.to(dtype=sglang_in.dtype))
    mini_local_with_sglang_weight = F.linear(mini_in, sglang_weight.to(dtype=mini_in.dtype))

    report_aligned_diff("local_partial_output", mini_local, sglang_local)
    report_aligned_diff("local_partial_output_same_weight", mini_local_with_sglang_weight, sglang_local)

    mini_local_aligned, sglang_local_aligned, stripped = maybe_strip_leading_bos(mini_local, sglang_local)
    print_alignment_note("local_partial_output_head", stripped)
    report_head_diff(
        "local_partial_output",
        mini_local_aligned[:, : NUM_Q_HEADS * HEAD_DIM],
        sglang_local_aligned[:, : NUM_Q_HEADS * HEAD_DIM],
    )

    report_aligned_diff("final_saved_output", mini_out_saved, sglang_out_saved)
    report_diff("mini_local_vs_saved_dump", mini_local, mini_out_saved)
    report_diff("sglang_local_vs_saved_dump", sglang_local, sglang_out_saved)

    mini_local_aligned, sglang_saved_aligned, stripped = maybe_strip_leading_bos(mini_local, sglang_out_saved)
    print_alignment_note("mini_local_vs_sglang_saved_dump", stripped)
    report_diff("mini_local_vs_sglang_saved_dump", mini_local_aligned, sglang_saved_aligned)

    mini_saved_aligned, sglang_local_aligned, stripped = maybe_strip_leading_bos(mini_out_saved, sglang_local)
    print_alignment_note("mini_saved_dump_vs_sglang_local", stripped)
    report_diff("mini_saved_dump_vs_sglang_local", mini_saved_aligned, sglang_local_aligned)

    report_top_diff("o_proj_weight", mini_weight, sglang_weight)

    mini_out_aligned, sglang_out_aligned, stripped = maybe_strip_leading_bos(mini_out_saved, sglang_out_saved)
    print_alignment_note("final_saved_output_head", stripped)
    report_head_diff(
        "final_saved_output",
        mini_out_aligned[:, : NUM_Q_HEADS * HEAD_DIM],
        sglang_out_aligned[:, : NUM_Q_HEADS * HEAD_DIM],
    )
    report_top_diff("final_saved_output", mini_out_aligned, sglang_out_aligned)

    print(
        "note: mini_out_saved / sglang_out_saved are final row-parallel outputs after all-reduce; "
        "mini_local / sglang_local are only local shard outputs. "
        "The HF full o_proj.weight has input dim 5120, so it cannot be directly multiplied "
        "with this rank's local 1280-d attn output without first reconstructing all TP input shards."
    )


if __name__ == "__main__":
    main()
