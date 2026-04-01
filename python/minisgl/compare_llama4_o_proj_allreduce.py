from __future__ import annotations

from pathlib import Path
import re

import torch
import torch.nn.functional as F


BASE_DIR = Path("tmp")
LAYER_ID = 0
NUM_Q_HEADS = 10
HEAD_DIM = 128


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
    coords_str = "(" + ", ".join(str(int(x.item())) for x in coords) + ")"
    lhs_val = lhs[tuple(int(x.item()) for x in coords)].item()
    rhs_val = rhs[tuple(int(x.item()) for x in coords)].item()
    print(
        f"{name}: shape={tuple(lhs.shape)}, "
        f"max_abs_diff={diff.max().item():.6g}, "
        f"mean_abs_diff={diff.mean().item():.6g}, "
        f"argmax_coord={coords_str}, "
        f"lhs_at_argmax={lhs_val}, rhs_at_argmax={rhs_val}"
    )


def report_aligned_diff(name: str, mini: torch.Tensor, sglang: torch.Tensor) -> None:
    mini_aligned, sglang_aligned, stripped = maybe_strip_leading_bos(mini, sglang)
    print_alignment_note(name, stripped)
    report_diff(name, mini_aligned, sglang_aligned)


def report_head_diff(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    lhs_h = lhs.float().reshape(-1, NUM_Q_HEADS, HEAD_DIM)
    rhs_h = rhs.float().reshape(-1, NUM_Q_HEADS, HEAD_DIM)
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


def find_ranked_files(prefix: str, stem: str) -> dict[int, Path]:
    pattern = re.compile(rf"{re.escape(prefix)}l{LAYER_ID}_rank(\d+)_{re.escape(stem)}\.pt$")
    matches: dict[int, Path] = {}
    for path in BASE_DIR.rglob(f"{prefix}l{LAYER_ID}_rank*_{stem}.pt"):
        match = pattern.match(path.name)
        if match:
            matches[int(match.group(1))] = path
    return dict(sorted(matches.items()))


def find_single_file(prefix: str, stem: str) -> Path | None:
    candidates = [
        BASE_DIR / f"{prefix}l{LAYER_ID}_{stem}.pt",
        BASE_DIR / f"{prefix}l{LAYER_ID}_reduced_{stem}.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    recursive_candidates = [
        f"{prefix}l{LAYER_ID}_{stem}.pt",
        f"{prefix}l{LAYER_ID}_reduced_{stem}.pt",
    ]
    for name in recursive_candidates:
        found = list(BASE_DIR.rglob(name))
        if found:
            return found[0]
    return None


def load_ranked_tensors(prefix: str, stem: str) -> dict[int, torch.Tensor]:
    files = find_ranked_files(prefix, stem)
    if not files:
        nearby = sorted(path.as_posix() for path in BASE_DIR.rglob(f"*rank*{stem}*.pt"))
        raise FileNotFoundError(
            f"No files found for pattern: {prefix}l{LAYER_ID}_rank*_{stem}.pt\n"
            f"Nearby matches:\n" + ("\n".join(nearby[:40]) if nearby else "(none)")
        )
    return {rank: torch.load(path, map_location="cpu") for rank, path in files.items()}


def load_optional_tensor(prefix: str, stem: str) -> torch.Tensor | None:
    path = find_single_file(prefix, stem)
    if path is None:
        return None
    return torch.load(path, map_location="cpu")


def verify_family(
    family: str,
    attn_inputs: dict[int, torch.Tensor],
    weights: dict[int, torch.Tensor],
    local_outputs: dict[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    print(f"=== {family} Local Reproduction ===")
    reconstructed: dict[int, torch.Tensor] = {}
    for rank in sorted(attn_inputs):
        attn_input = attn_inputs[rank]
        weight = weights[rank]
        saved_local = local_outputs[rank]
        constructed = F.linear(attn_input, weight.to(dtype=attn_input.dtype))
        reconstructed[rank] = constructed
        report_diff(f"{family}_rank{rank}_constructed_vs_saved_local", constructed, saved_local)
    return reconstructed


def sum_ranked_tensors(tensors: dict[int, torch.Tensor]) -> torch.Tensor:
    ranks = sorted(tensors)
    total = tensors[ranks[0]].clone()
    for rank in ranks[1:]:
        total = total + tensors[rank]
    return total


def sum_ranked_tensors_fp32(tensors: dict[int, torch.Tensor]) -> torch.Tensor:
    ranks = sorted(tensors)
    total = tensors[ranks[0]].float()
    for rank in ranks[1:]:
        total = total + tensors[rank].float()
    return total


def main() -> None:
    mini_inputs = load_ranked_tensors("", "attn_output")
    mini_weights = load_ranked_tensors("", "o_proj_weight")
    mini_locals = load_ranked_tensors("", "o_proj_output")

    sglang_inputs = load_ranked_tensors("sglang_", "attn_output")
    sglang_weights = load_ranked_tensors("sglang_", "o_proj_weight")
    sglang_locals = load_ranked_tensors("sglang_", "o_proj_output")

    mini_reduced = load_optional_tensor("", "o_proj_output")
    sglang_reduced = load_optional_tensor("sglang_", "o_proj_output")

    print(f"mini_ranks=={sorted(mini_inputs)}")
    print(f"sglang_ranks=={sorted(sglang_inputs)}")

    mini_reconstructed = verify_family("mini", mini_inputs, mini_weights, mini_locals)
    sglang_reconstructed = verify_family("sglang", sglang_inputs, sglang_weights, sglang_locals)

    print("=== Cross-Framework Per-Rank ===")
    for rank in sorted(mini_inputs):
        report_aligned_diff(f"rank{rank}_attn_input", mini_inputs[rank], sglang_inputs[rank])
        report_diff(f"rank{rank}_o_proj_weight", mini_weights[rank], sglang_weights[rank])
        report_aligned_diff(f"rank{rank}_saved_local_output", mini_locals[rank], sglang_locals[rank])

    mini_local_sum = sum_ranked_tensors(mini_locals)
    sglang_local_sum = sum_ranked_tensors(sglang_locals)
    mini_reconstructed_sum = sum_ranked_tensors(mini_reconstructed)
    sglang_reconstructed_sum = sum_ranked_tensors(sglang_reconstructed)
    mini_local_sum_fp32 = sum_ranked_tensors_fp32(mini_locals)
    sglang_local_sum_fp32 = sum_ranked_tensors_fp32(sglang_locals)
    mini_reconstructed_sum_fp32 = sum_ranked_tensors_fp32(mini_reconstructed)
    sglang_reconstructed_sum_fp32 = sum_ranked_tensors_fp32(sglang_reconstructed)

    print("=== Manual All-Reduce Reconstruction ===")
    report_diff("mini_manual_sum_vs_reconstructed_sum", mini_local_sum, mini_reconstructed_sum)
    report_diff("sglang_manual_sum_vs_reconstructed_sum", sglang_local_sum, sglang_reconstructed_sum)
    report_diff("mini_manual_sum_fp32_vs_reconstructed_sum_fp32", mini_local_sum_fp32, mini_reconstructed_sum_fp32)
    report_diff("sglang_manual_sum_fp32_vs_reconstructed_sum_fp32", sglang_local_sum_fp32, sglang_reconstructed_sum_fp32)

    if mini_reduced is not None:
        report_diff("mini_manual_sum_vs_saved_reduced", mini_local_sum, mini_reduced)
        report_diff("mini_manual_sum_fp32_vs_saved_reduced", mini_local_sum_fp32, mini_reduced.float())
    else:
        print("mini_saved_reduced: not found")

    if sglang_reduced is not None:
        report_diff("sglang_manual_sum_vs_saved_reduced", sglang_local_sum, sglang_reduced)
        report_diff("sglang_manual_sum_fp32_vs_saved_reduced", sglang_local_sum_fp32, sglang_reduced.float())
        report_aligned_diff("mini_local_vs_sglang_saved_reduced", mini_local_sum, sglang_reduced)
    else:
        print("sglang_saved_reduced: not found")

    print("=== Cross-Framework Reduced Output ===")
    report_aligned_diff("manual_allreduced_output", mini_local_sum, sglang_local_sum)
    report_aligned_diff("manual_allreduced_output_fp32", mini_local_sum_fp32, sglang_local_sum_fp32)

    mini_sum_aligned, sglang_sum_aligned, stripped = maybe_strip_leading_bos(mini_local_sum, sglang_local_sum)
    print_alignment_note("manual_allreduced_output_head", stripped)
    report_head_diff(
        "manual_allreduced_output",
        mini_sum_aligned[:, : NUM_Q_HEADS * HEAD_DIM],
        sglang_sum_aligned[:, : NUM_Q_HEADS * HEAD_DIM],
    )
    report_top_diff("manual_allreduced_output", mini_sum_aligned, sglang_sum_aligned)

    mini_sum_fp32_aligned, sglang_sum_fp32_aligned, stripped = maybe_strip_leading_bos(
        mini_local_sum_fp32, sglang_local_sum_fp32
    )
    print_alignment_note("manual_allreduced_output_fp32_head", stripped)
    report_head_diff(
        "manual_allreduced_output_fp32",
        mini_sum_fp32_aligned[:, : NUM_Q_HEADS * HEAD_DIM],
        sglang_sum_fp32_aligned[:, : NUM_Q_HEADS * HEAD_DIM],
    )
    report_top_diff("manual_allreduced_output_fp32", mini_sum_fp32_aligned, sglang_sum_fp32_aligned)

    if mini_reduced is not None and sglang_reduced is not None:
        report_aligned_diff("saved_reduced_output", mini_reduced, sglang_reduced)
        mini_reduced_aligned, sglang_reduced_aligned, stripped = maybe_strip_leading_bos(
            mini_reduced, sglang_reduced
        )
        print_alignment_note("saved_reduced_output_head", stripped)
        report_head_diff(
            "saved_reduced_output",
            mini_reduced_aligned[:, : NUM_Q_HEADS * HEAD_DIM],
            sglang_reduced_aligned[:, : NUM_Q_HEADS * HEAD_DIM],
        )

        report_aligned_diff("manual_sum_vs_saved_reduced_cross", mini_local_sum, sglang_reduced)
        report_aligned_diff("saved_reduced_vs_manual_sum_cross", mini_reduced, sglang_local_sum)
        report_aligned_diff("manual_sum_fp32_vs_saved_reduced_cross", mini_local_sum_fp32, sglang_reduced.float())
        report_aligned_diff("saved_reduced_vs_manual_sum_fp32_cross", mini_reduced.float(), sglang_local_sum_fp32)


if __name__ == "__main__":
    main()
