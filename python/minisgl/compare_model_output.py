from __future__ import annotations

from pathlib import Path

import torch


MINI_PATH = Path("tmp/model_output.pt")
SGLANG_PATH = Path("tmp/sglang_model_output.pt")


def maybe_strip_leading_bos(
    mini: torch.Tensor, sglang: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if mini.ndim > 0 and sglang.ndim > 0 and mini.shape[0] == sglang.shape[0] + 1:
        return mini[1:], sglang, True
    return mini, sglang, False


def report_diff(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    diff = (lhs.float() - rhs.float()).abs()
    flat_idx = int(diff.argmax().item())
    coords = tuple(int(x.item()) for x in torch.unravel_index(torch.tensor(flat_idx), diff.shape))
    print(
        f"{name}: shape={tuple(lhs.shape)}, "
        f"max_abs_diff={diff.max().item():.6g}, "
        f"mean_abs_diff={diff.mean().item():.6g}, "
        f"argmax_coord={coords}, "
        f"lhs_at_argmax={lhs[coords].item()}, rhs_at_argmax={rhs[coords].item()}"
    )


def report_token_diff(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    diff = (lhs.float() - rhs.float()).abs()
    print(f"{name}_per_token_max=={diff.amax(dim=1)}")
    print(f"{name}_per_token_mean=={diff.mean(dim=1)}")


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
            f"lhs={lhs[row, col].item()}, rhs={rhs[row, col].item()}"
        )


def report_top_magnitude(name: str, tensor: torch.Tensor, limit: int = 10) -> None:
    flat = tensor.float().abs().flatten()
    k = min(limit, flat.numel())
    vals, idxs = torch.topk(flat, k=k)
    last_dim = tensor.shape[-1]
    print(f"top_{name}_magnitude:")
    for rank, (val, idx) in enumerate(zip(vals.tolist(), idxs.tolist()), start=1):
        row = idx // last_dim
        col = idx % last_dim
        print(f"{rank}: row={row}, col={col}, abs_val={val:.6g}, value={tensor[row, col].item()}")


def main() -> None:
    mini = torch.load(MINI_PATH, map_location="cpu")
    sglang = torch.load(SGLANG_PATH, map_location="cpu")

    print(f"mini.shape=={mini.shape}, dtype=={mini.dtype}")
    print(f"sglang.shape=={sglang.shape}, dtype=={sglang.dtype}")

    mini_aligned, sglang_aligned, stripped = maybe_strip_leading_bos(mini, sglang)
    if stripped:
        print("model_output: stripped leading BOS row from mini before comparison")

    report_diff("model_output", mini_aligned, sglang_aligned)
    report_token_diff("model_output", mini_aligned, sglang_aligned)
    report_top_diff("model_output", mini_aligned, sglang_aligned, limit=10)
    report_top_magnitude("mini_model_output", mini_aligned, limit=10)
    report_top_magnitude("sglang_model_output", sglang_aligned, limit=10)


if __name__ == "__main__":
    main()
