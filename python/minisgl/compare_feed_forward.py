from __future__ import annotations

from pathlib import Path

import torch


BASE_DIR = Path("tmp")
LAYER_ID = 0


def maybe_strip_leading_bos(
    mini: torch.Tensor, sglang: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if mini.ndim > 0 and sglang.ndim > 0 and mini.shape[0] == sglang.shape[0] + 1:
        return mini[1:], sglang, True
    return mini, sglang, False


def print_alignment_note(name: str, stripped: bool) -> None:
    if stripped:
        print(f"{name}: stripped leading BOS row from mini before comparison")


def report_diff(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[torch.Tensor, int]:
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
    return diff, flat_idx


def report_aligned_diff(
    name: str, mini: torch.Tensor, sglang: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    mini_aligned, sglang_aligned, stripped = maybe_strip_leading_bos(mini, sglang)
    print_alignment_note(name, stripped)
    report_diff(name, mini_aligned, sglang_aligned)
    return mini_aligned, sglang_aligned


def report_token_diff(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    diff = (lhs.float() - rhs.float()).abs()
    token_max = diff.amax(dim=1)
    token_mean = diff.mean(dim=1)
    print(f"{name}_per_token_max=={token_max}")
    print(f"{name}_per_token_mean=={token_mean}")


def report_l2norm_diff(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    lhs_norm = lhs.float().norm(dim=1)
    rhs_norm = rhs.float().norm(dim=1)
    report_diff(f"{name}_l2norm", lhs_norm, rhs_norm)


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


def main() -> None:
    mini_input = torch.load(
        BASE_DIR / f"l{LAYER_ID}_post_attention_layernorm_output.pt", map_location="cpu"
    )
    sglang_input = torch.load(
        BASE_DIR / f"sglang_l{LAYER_ID}_post_attention_layernorm_output.pt",
        map_location="cpu",
    )
    mini_output = torch.load(BASE_DIR / f"l{LAYER_ID}_after_mlp_hidden_states.pt", map_location="cpu")
    sglang_output = torch.load(
        BASE_DIR / f"sglang_l{LAYER_ID}_after_mlp_hidden_states.pt", map_location="cpu"
    )

    print(f"mini_input.shape=={mini_input.shape}, dtype=={mini_input.dtype}")
    print(f"sglang_input.shape=={sglang_input.shape}, dtype=={sglang_input.dtype}")
    print(f"mini_output.shape=={mini_output.shape}, dtype=={mini_output.dtype}")
    print(f"sglang_output.shape=={sglang_output.shape}, dtype=={sglang_output.dtype}")

    mini_input_aligned, sglang_input_aligned = report_aligned_diff(
        "feed_forward_input", mini_input, sglang_input
    )
    report_token_diff("feed_forward_input", mini_input_aligned, sglang_input_aligned)
    report_l2norm_diff("feed_forward_input", mini_input_aligned, sglang_input_aligned)

    mini_output_aligned, sglang_output_aligned = report_aligned_diff(
        "feed_forward_output", mini_output, sglang_output
    )
    report_token_diff("feed_forward_output", mini_output_aligned, sglang_output_aligned)
    report_l2norm_diff("feed_forward_output", mini_output_aligned, sglang_output_aligned)
    report_top_diff("feed_forward_output", mini_output_aligned, sglang_output_aligned)

    mini_delta = mini_output_aligned.float() - mini_input_aligned.float()
    sglang_delta = sglang_output_aligned.float() - sglang_input_aligned.float()
    report_diff("feed_forward_delta", mini_delta, sglang_delta)
    report_token_diff("feed_forward_delta", mini_delta, sglang_delta)
    report_l2norm_diff("feed_forward_delta", mini_delta, sglang_delta)
    report_top_diff("feed_forward_delta", mini_delta, sglang_delta)


if __name__ == "__main__":
    main()
