from __future__ import annotations

from pathlib import Path

import torch


BASE_DIR = Path("tmp")
LAYER_ID = 1
NUM_Q_HEADS = 10
NUM_KV_HEADS = 2
HEAD_DIM = 128


def maybe_strip_leading_bos(
    mini: torch.Tensor, sglang: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if mini.ndim > 0 and sglang.ndim > 0 and mini.shape[0] == sglang.shape[0] + 1:
        return mini[1:], sglang, True
    return mini, sglang, False


def normalize_pair_shapes(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if lhs.shape == rhs.shape:
        return lhs, rhs

    if lhs.ndim == 3 and rhs.ndim == 2 and lhs.shape[0] == rhs.shape[0]:
        return lhs.reshape(lhs.shape[0], -1), rhs

    if lhs.ndim == 2 and rhs.ndim == 3 and lhs.shape[0] == rhs.shape[0]:
        return lhs, rhs.reshape(rhs.shape[0], -1)

    return lhs, rhs


def print_alignment_note(name: str, stripped: bool) -> None:
    if stripped:
        print(f"{name}: stripped leading BOS row from mini before comparison")


def report_diff(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
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
    return diff, coord_tuple


def report_aligned_diff(
    name: str, mini: torch.Tensor, sglang: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    mini_aligned, sglang_aligned, stripped = maybe_strip_leading_bos(mini, sglang)
    mini_aligned, sglang_aligned = normalize_pair_shapes(mini_aligned, sglang_aligned)
    print_alignment_note(name, stripped)
    report_diff(name, mini_aligned, sglang_aligned)
    return mini_aligned, sglang_aligned


def report_token_diff(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    diff = (lhs.float() - rhs.float()).abs()
    print(f"{name}_per_token_max=={diff.amax(dim=1)}")
    print(f"{name}_per_token_mean=={diff.mean(dim=1)}")


def report_head_diff(name: str, lhs: torch.Tensor, rhs: torch.Tensor, num_heads: int) -> None:
    lhs_h = lhs.float().reshape(-1, num_heads, HEAD_DIM)
    rhs_h = rhs.float().reshape(-1, num_heads, HEAD_DIM)
    diff = (lhs_h - rhs_h).abs().amax(dim=(0, 2))
    print(f"{name}_head_diff=={diff}")


def report_top_diff(name: str, lhs: torch.Tensor, rhs: torch.Tensor, limit: int = 10) -> None:
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


def load_pair(stem: str) -> tuple[torch.Tensor, torch.Tensor]:
    mini = torch.load(BASE_DIR / f"l{LAYER_ID}_{stem}.pt", map_location="cpu")
    sglang = torch.load(BASE_DIR / f"sglang_l{LAYER_ID}_{stem}.pt", map_location="cpu")
    return mini, sglang


def section(title: str) -> None:
    print(f"=== {title} ===")


def main() -> None:
    stage_specs = [
        ("Input LayerNorm Output", "before_attn_hidden_states", None),
        ("QKV Proj Output", "attn_after_qkv_proj", None),
        ("RoPE Q", "after_rope_q", ("head", NUM_Q_HEADS)),
        ("RoPE K", "after_rope_k", ("head", NUM_KV_HEADS)),
        ("RoPE V", "after_rope_v", ("head", NUM_KV_HEADS)),
        ("QKNorm Q", "after_qknorm_q", ("head", NUM_Q_HEADS)),
        ("QKNorm K", "after_qknorm_k", ("head", NUM_KV_HEADS)),
        ("QKNorm V", "after_qknorm_v", ("head", NUM_KV_HEADS)),
        ("Before Attention Backend Q", "before_attn_backend_q", ("head", NUM_Q_HEADS)),
        ("Before Attention Backend K", "before_attn_backend_k", ("head", NUM_KV_HEADS)),
        ("Before Attention Backend V", "before_attn_backend_v", ("head", NUM_KV_HEADS)),
        ("Attention Core Output", "attn_after_attention", None),
        ("Attention Final Output", "reduced_o_proj_output", None),
        ("Post-Attn LayerNorm Residual In", "before_post_attn_layernorm_residual", None),
        ("Post-Attn LayerNorm Output", "post_attention_layernorm_output", None),
        ("Post-Attn LayerNorm Residual Out", "post_attention_layernorm_residual_output", None),
        ("Feed Forward Output", "after_mlp_hidden_states", None),
    ]

    for title, stem, extra in stage_specs:
        section(title)
        mini, sglang = load_pair(stem)
        mini_aligned, sglang_aligned = report_aligned_diff(stem, mini, sglang)
        report_token_diff(stem, mini_aligned, sglang_aligned)
        if extra is not None and extra[0] == "head":
            report_head_diff(stem, mini_aligned, sglang_aligned, extra[1])
        report_top_diff(stem, mini_aligned, sglang_aligned)

    section("Stage Deltas")
    delta_specs = [
        ("attn_delta", "before_attn_hidden_states", "reduced_o_proj_output"),
        ("post_attn_layernorm_delta", "reduced_o_proj_output", "post_attention_layernorm_output"),
        ("feed_forward_delta", "post_attention_layernorm_output", "after_mlp_hidden_states"),
    ]
    for name, in_stem, out_stem in delta_specs:
        mini_in, sglang_in = load_pair(in_stem)
        mini_out, sglang_out = load_pair(out_stem)
        mini_in_aligned, sglang_in_aligned, _ = maybe_strip_leading_bos(mini_in, sglang_in)
        mini_out_aligned, sglang_out_aligned, _ = maybe_strip_leading_bos(mini_out, sglang_out)
        mini_in_aligned, sglang_in_aligned = normalize_pair_shapes(mini_in_aligned, sglang_in_aligned)
        mini_out_aligned, sglang_out_aligned = normalize_pair_shapes(mini_out_aligned, sglang_out_aligned)
        mini_delta = mini_out_aligned.float() - mini_in_aligned.float()
        sglang_delta = sglang_out_aligned.float() - sglang_in_aligned.float()
        report_diff(name, mini_delta, sglang_delta)
        report_token_diff(name, mini_delta, sglang_delta)
        report_top_diff(name, mini_delta, sglang_delta)


if __name__ == "__main__":
    main()
