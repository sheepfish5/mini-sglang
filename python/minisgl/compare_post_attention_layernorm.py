from __future__ import annotations

from pathlib import Path

import torch


BASE_DIR = Path("tmp")
LAYER_ID = 0
HEAD_DIM = 128
NUM_Q_HEADS = 10
EPS = 1e-6


def maybe_strip_leading_bos(mini: torch.Tensor, sglang: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if mini.ndim > 0 and sglang.ndim > 0 and mini.shape[0] == sglang.shape[0] + 1:
        return mini[1:], sglang, True
    return mini, sglang, False


def print_alignment_note(name: str, stripped: bool) -> None:
    if stripped:
        print(f"{name}: stripped leading BOS row from mini before comparison")


def report_diff(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    diff = (lhs.float() - rhs.float()).abs()
    flat_idx = int(diff.argmax().item())
    coords = list(torch.unravel_index(torch.tensor(flat_idx), diff.shape))
    coord_tuple = tuple(int(x.item()) for x in coords)
    print(
        f"{name}: shape={tuple(lhs.shape)}, "
        f"max_abs_diff={diff.max().item():.6g}, "
        f"mean_abs_diff={diff.mean().item():.6g}, "
        f"argmax_coord={coord_tuple}, "
        f"lhs_at_argmax={lhs[coord_tuple].item()}, rhs_at_argmax={rhs[coord_tuple].item()}"
    )


def report_aligned_diff(name: str, mini: torch.Tensor, sglang: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mini_aligned, sglang_aligned, stripped = maybe_strip_leading_bos(mini, sglang)
    print_alignment_note(name, stripped)
    report_diff(name, mini_aligned, sglang_aligned)
    return mini_aligned, sglang_aligned


def report_head_diff(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    lhs_h = lhs.float().reshape(-1, NUM_Q_HEADS, HEAD_DIM)
    rhs_h = rhs.float().reshape(-1, NUM_Q_HEADS, HEAD_DIM)
    diff = (lhs_h - rhs_h).abs().amax(dim=(0, 2))
    print(f"{name}_head_diff=={diff}")


def report_token_diff(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    diff = (lhs.float() - rhs.float()).abs()
    token_max = diff.amax(dim=1)
    token_mean = diff.mean(dim=1)
    print(f"{name}_per_token_max=={token_max}")
    print(f"{name}_per_token_mean=={token_mean}")


def report_top_diff(name: str, lhs: torch.Tensor, rhs: torch.Tensor, limit: int = 20) -> None:
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


def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_fp32 = x.float()
    weight_fp32 = weight.float()
    rms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
    out = x_fp32 * rms * weight_fp32
    return out.to(dtype=x.dtype)


def main() -> None:
    mini_x = torch.load(BASE_DIR / f"l{LAYER_ID}_reduced_o_proj_output.pt", map_location="cpu")
    sglang_x = torch.load(BASE_DIR / f"sglang_l{LAYER_ID}_reduced_o_proj_output.pt", map_location="cpu")
    mini_residual_in = torch.load(
        BASE_DIR / f"l{LAYER_ID}_before_post_attn_layernorm_residual.pt", map_location="cpu"
    )
    sglang_residual_in = torch.load(
        BASE_DIR / f"sglang_l{LAYER_ID}_before_post_attn_layernorm_residual.pt", map_location="cpu"
    )
    mini_weight = torch.load(BASE_DIR / f"l{LAYER_ID}_post_attn_layernorm_weight.pt", map_location="cpu")
    sglang_weight = torch.load(BASE_DIR / f"sglang_l{LAYER_ID}_post_attn_layernorm_weight.pt", map_location="cpu")
    mini_output = torch.load(BASE_DIR / f"l{LAYER_ID}_post_attention_layernorm_output.pt", map_location="cpu")
    sglang_output = torch.load(BASE_DIR / f"sglang_l{LAYER_ID}_post_attention_layernorm_output.pt", map_location="cpu")
    mini_residual_out = torch.load(
        BASE_DIR / f"l{LAYER_ID}_post_attention_layernorm_residual_output.pt", map_location="cpu"
    )
    sglang_residual_out = torch.load(
        BASE_DIR / f"sglang_l{LAYER_ID}_post_attention_layernorm_residual_output.pt", map_location="cpu"
    )

    print(f"mini_x.shape=={mini_x.shape}, dtype=={mini_x.dtype}")
    print(f"sglang_x.shape=={sglang_x.shape}, dtype=={sglang_x.dtype}")
    print(f"mini_residual_in.shape=={mini_residual_in.shape}, dtype=={mini_residual_in.dtype}")
    print(f"sglang_residual_in.shape=={sglang_residual_in.shape}, dtype=={sglang_residual_in.dtype}")
    print(f"mini_weight.shape=={mini_weight.shape}, dtype=={mini_weight.dtype}")
    print(f"sglang_weight.shape=={sglang_weight.shape}, dtype=={sglang_weight.dtype}")
    print(f"mini_output.shape=={mini_output.shape}, dtype=={mini_output.dtype}")
    print(f"sglang_output.shape=={sglang_output.shape}, dtype=={sglang_output.dtype}")
    print(f"mini_residual_out.shape=={mini_residual_out.shape}, dtype=={mini_residual_out.dtype}")
    print(f"sglang_residual_out.shape=={sglang_residual_out.shape}, dtype=={sglang_residual_out.dtype}")

    mini_x_aligned, sglang_x_aligned = report_aligned_diff("post_attn_layernorm_x", mini_x, sglang_x)
    mini_residual_in_aligned, sglang_residual_in_aligned = report_aligned_diff(
        "post_attn_layernorm_residual_in", mini_residual_in, sglang_residual_in
    )
    report_diff("post_attn_layernorm_weight", mini_weight, sglang_weight)

    mini_residual_out_aligned, sglang_residual_out_aligned = report_aligned_diff(
        "post_attn_layernorm_residual_out", mini_residual_out, sglang_residual_out
    )
    mini_output_aligned, sglang_output_aligned = report_aligned_diff(
        "post_attn_layernorm_output", mini_output, sglang_output
    )

    report_head_diff(
        "post_attn_layernorm_output",
        mini_output_aligned[:, : NUM_Q_HEADS * HEAD_DIM],
        sglang_output_aligned[:, : NUM_Q_HEADS * HEAD_DIM],
    )
    report_token_diff("post_attn_layernorm_output", mini_output_aligned, sglang_output_aligned)
    report_top_diff("post_attn_layernorm_output", mini_output_aligned, sglang_output_aligned)

    mini_added = mini_x_aligned.float() + mini_residual_in_aligned.float()
    sglang_added = sglang_x_aligned.float() + sglang_residual_in_aligned.float()
    report_diff("post_attn_layernorm_added_input", mini_added, sglang_added)
    report_token_diff("post_attn_layernorm_added_input", mini_added, sglang_added)
    report_top_diff("post_attn_layernorm_added_input", mini_added, sglang_added)

    report_diff("post_attn_layernorm_added_input_vs_saved_residual_out_mini", mini_added, mini_residual_out_aligned.float())
    report_diff(
        "post_attn_layernorm_added_input_vs_saved_residual_out_sglang",
        sglang_added,
        sglang_residual_out_aligned.float(),
    )

    mini_ref = rmsnorm_ref(mini_added.to(mini_output_aligned.dtype), mini_weight, EPS)
    sglang_ref = rmsnorm_ref(sglang_added.to(sglang_output_aligned.dtype), sglang_weight, EPS)
    report_diff("mini_post_attn_layernorm_ref_vs_saved_output", mini_ref, mini_output_aligned)
    report_diff("sglang_post_attn_layernorm_ref_vs_saved_output", sglang_ref, sglang_output_aligned)
    report_aligned_diff("post_attn_layernorm_ref_output_cross", mini_ref, sglang_ref)

    mini_x_times4 = mini_x_aligned.float() * 4.0
    mini_added_times4 = mini_x_times4 + mini_residual_in_aligned.float()
    mini_ref_times4 = rmsnorm_ref(mini_added_times4.to(mini_output_aligned.dtype), mini_weight, EPS)
    report_diff("post_attn_layernorm_added_input_mini_x4_vs_sglang_added_input", mini_added_times4, sglang_added)
    report_diff("mini_post_attn_layernorm_ref_x4_vs_saved_output", mini_ref_times4, mini_output_aligned)
    report_diff("mini_post_attn_layernorm_ref_x4_vs_sglang_saved_output", mini_ref_times4, sglang_output_aligned)
    report_diff("mini_post_attn_layernorm_ref_x4_vs_sglang_ref", mini_ref_times4, sglang_ref)
    report_head_diff(
        "mini_post_attn_layernorm_ref_x4_vs_sglang_saved_output",
        mini_ref_times4[:, : NUM_Q_HEADS * HEAD_DIM],
        sglang_output_aligned[:, : NUM_Q_HEADS * HEAD_DIM],
    )
    report_top_diff("mini_post_attn_layernorm_ref_x4_vs_sglang_saved_output", mini_ref_times4, sglang_output_aligned)

    mini_ref_delta = mini_ref.float() - mini_added
    sglang_ref_delta = sglang_ref.float() - sglang_added
    report_diff("post_attn_layernorm_ref_delta_cross", mini_ref_delta, sglang_ref_delta)
    report_top_diff("post_attn_layernorm_ref_delta_cross", mini_ref_delta, sglang_ref_delta)

    print(
        "note: RMSNormFused.forward(x, residual) first accumulates residual, then applies RMSNorm. "
        "This script checks x, residual, accumulated residual output, norm weight, and a reference RMSNorm."
    )


if __name__ == "__main__":
    main()
