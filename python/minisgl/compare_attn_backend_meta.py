from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


ATTN_DEBUG_DIR = Path("tmp/attn_debug")
LAYERS = (0, 1)
HEAD_DIM = 128
NUM_Q_HEADS = 10
NUM_KV_HEADS = 2


def maybe_strip_leading_bos(mini: torch.Tensor, sglang: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if mini.ndim == 0 or sglang.ndim == 0:
        return mini, sglang, False
    if mini.shape[0] == sglang.shape[0] + 1:
        return mini[1:], sglang, True
    return mini, sglang, False


def print_alignment_note(name: str, stripped: bool) -> None:
    if stripped:
        print(f"{name}: stripped leading BOS row from mini before comparison")


def load_pt(path: Path) -> Any:
    return torch.load(path, map_location="cpu")


def as_tensor(x: Any) -> torch.Tensor | None:
    return x if isinstance(x, torch.Tensor) else None


def flatten_last_two(x: torch.Tensor) -> torch.Tensor:
    if x.ndim >= 3:
        return x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1])
    return x


def report_diff(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    lhs = lhs.float()
    rhs = rhs.float()
    if lhs.shape != rhs.shape:
        print(f"{name}: shape mismatch {tuple(lhs.shape)} vs {tuple(rhs.shape)}")
        return
    diff = (lhs - rhs).abs()
    print(
        f"{name}: shape={tuple(lhs.shape)}, "
        f"max_abs_diff={diff.max().item():.6g}, "
        f"mean_abs_diff={diff.mean().item():.6g}"
    )


def report_head_diff(name: str, lhs: torch.Tensor, rhs: torch.Tensor, num_heads: int) -> None:
    lhs = flatten_last_two(lhs).float().view(-1, num_heads, HEAD_DIM)
    rhs = flatten_last_two(rhs).float().view(-1, num_heads, HEAD_DIM)
    diff = (lhs - rhs).abs().amax(dim=(0, 2))
    print(f"{name}_head_diff=={diff}")


def report_tensor_or_scalar(name: str, lhs: Any, rhs: Any) -> None:
    lt = as_tensor(lhs)
    rt = as_tensor(rhs)
    if lt is not None and rt is not None:
        report_diff(name, lt, rt)
    else:
        print(f"{name}: mini={lhs!r}, sglang={rhs!r}")


def report_aligned_diff(name: str, mini: torch.Tensor, sglang: torch.Tensor) -> None:
    mini_aligned, sglang_aligned, stripped = maybe_strip_leading_bos(mini, sglang)
    print_alignment_note(name, stripped)
    report_diff(name, mini_aligned, sglang_aligned)


def report_aligned_head_diff(name: str, mini: torch.Tensor, sglang: torch.Tensor, num_heads: int) -> None:
    mini_aligned, sglang_aligned, stripped = maybe_strip_leading_bos(mini, sglang)
    print_alignment_note(f"{name}_head", stripped)
    report_head_diff(name, mini_aligned, sglang_aligned, num_heads)


def mini_req_summary(meta: dict[str, Any]) -> dict[str, Any]:
    reqs = meta["reqs"]
    padded_reqs = meta["padded_reqs"]
    return {
        "table_idx": torch.tensor([r["table_idx"] for r in reqs], dtype=torch.int64),
        "cached_len": torch.tensor([r["cached_len"] for r in reqs], dtype=torch.int32),
        "device_len": torch.tensor([r["device_len"] for r in reqs], dtype=torch.int32),
        "extend_len": torch.tensor([r["extend_len"] for r in reqs], dtype=torch.int32),
        "padded_table_idx": torch.tensor([r["table_idx"] for r in padded_reqs], dtype=torch.int64),
        "padded_cached_len": torch.tensor([r["cached_len"] for r in padded_reqs], dtype=torch.int32),
        "padded_device_len": torch.tensor([r["device_len"] for r in padded_reqs], dtype=torch.int32),
        "padded_extend_len": torch.tensor([r["extend_len"] for r in padded_reqs], dtype=torch.int32),
    }


def expand_per_query(prefix_lens: torch.Tensor, extend_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    prefix_parts = []
    full_parts = []
    for prefix_len, extend_len in zip(prefix_lens.tolist(), extend_lens.tolist()):
        prefix_parts.append(torch.full((extend_len,), int(prefix_len), dtype=torch.int32))
        full_parts.append(torch.arange(1, extend_len + 1, dtype=torch.int32) + int(prefix_len))
    if not prefix_parts:
        empty = torch.empty(0, dtype=torch.int32)
        return empty, empty
    return torch.cat(prefix_parts), torch.cat(full_parts)


def compare_visibility_semantics(layer_id: int, mini: dict[str, Any], sglang: dict[str, Any]) -> None:
    print(f"=== Layer {layer_id} Query Visibility Semantics ===")
    mini_req = mini_req_summary(mini)

    mini_prefix_per_q, mini_full_per_q = expand_per_query(
        mini_req["cached_len"], mini_req["extend_len"]
    )
    sg_prefix_per_q, sg_full_per_q = expand_per_query(
        sglang["extend_prefix_lens"], sglang["extend_seq_lens"]
    )

    report_aligned_diff("prefix_visible_kv_len_per_query", mini_prefix_per_q, sg_prefix_per_q)
    report_aligned_diff("full_visible_kv_len_per_query", mini_full_per_q, sg_full_per_q)
    print(f"mini_prefix_visible_kv_len_per_query=={mini_prefix_per_q}")
    print(f"sglang_prefix_visible_kv_len_per_query=={sg_prefix_per_q}")
    print(f"mini_full_visible_kv_len_per_query=={mini_full_per_q}")
    print(f"sglang_full_visible_kv_len_per_query=={sg_full_per_q}")

    mini_indices = mini["attn_metadata"].get("indices")
    sg_kv_indices = sglang["attn_metadata"].get("kv_indices")
    sg_kv_indptr = sglang["attn_metadata"].get("kv_indptr")
    if isinstance(mini_indices, torch.Tensor):
        print(f"mini_all_kv_cache_indices=={mini_indices}")
    if isinstance(sg_kv_indices, torch.Tensor) and isinstance(sg_kv_indptr, torch.Tensor):
        print(f"sglang_prefix_kv_indptr=={sg_kv_indptr}")
        print(f"sglang_prefix_kv_indices=={sg_kv_indices}")


def compare_common_fields(layer_id: int, mini: dict[str, Any], sglang: dict[str, Any]) -> None:
    print(f"=== Layer {layer_id} Common Fields ===")
    print(
        f"backend_type: mini={mini['backend_type']}, "
        f"sglang={sglang['backend_type']}"
    )
    print(
        f"metadata_type: mini={mini['metadata_type']}, "
        f"sglang={sglang['attn_metadata'].get('type')}"
    )

    mini_pos, sglang_pos, stripped = maybe_strip_leading_bos(mini["positions"], sglang["positions"])
    print_alignment_note("positions", stripped)
    report_tensor_or_scalar("positions", mini_pos, sglang_pos)
    report_aligned_diff("q", flatten_last_two(mini["q"]), flatten_last_two(sglang["q"]))
    report_aligned_diff("k", flatten_last_two(mini["k"]), flatten_last_two(sglang["k"]))
    report_aligned_diff("v", flatten_last_two(mini["v"]), flatten_last_two(sglang["v"]))
    report_aligned_head_diff("q", mini["q"], sglang["q"], NUM_Q_HEADS)
    report_aligned_head_diff("k", mini["k"], sglang["k"], NUM_KV_HEADS)
    report_aligned_head_diff("v", mini["v"], sglang["v"], NUM_KV_HEADS)


def compare_batch_semantics(layer_id: int, mini: dict[str, Any], sglang: dict[str, Any]) -> None:
    print(f"=== Layer {layer_id} Batch Semantics ===")
    mini_req = mini_req_summary(mini)

    report_tensor_or_scalar("mini_phase_vs_sglang_mode", mini["phase"], sglang["forward_mode"])
    report_diff("extend_len", mini_req["extend_len"], sglang["extend_seq_lens"])
    report_diff("cached_len", mini_req["cached_len"], sglang["extend_prefix_lens"])
    report_diff("device_len_vs_seq_lens", mini_req["device_len"], sglang["seq_lens"])
    report_diff("table_idx_vs_req_pool_indices", mini_req["table_idx"], sglang["req_pool_indices"])
    report_tensor_or_scalar("out_loc", mini["out_loc"], None)

    mini_seq_lens_cpu = mini["attn_metadata"].get("seq_lens_cpu")
    mini_cu_q = mini["attn_metadata"].get("cu_seqlens_q_cpu")
    mini_cu_k = mini["attn_metadata"].get("cu_seqlens_k_cpu")
    if mini_seq_lens_cpu is not None:
        report_diff("mini_seq_lens_cpu_vs_sglang_seq_lens_cpu", mini_seq_lens_cpu, sglang["seq_lens_cpu"])
    if mini_cu_q is not None and sglang["extend_seq_lens"] is not None:
        sglang_cu_q = torch.cat(
            [
                torch.zeros(1, dtype=sglang["extend_seq_lens"].dtype),
                sglang["extend_seq_lens"].cumsum(dim=0),
            ]
        )
        report_diff("mini_cu_seqlens_q_vs_sglang_extend_cumsum", mini_cu_q, sglang_cu_q)
    if mini_cu_k is not None and sglang["seq_lens"] is not None:
        sglang_cu_k = torch.cat(
            [
                torch.zeros(1, dtype=sglang["seq_lens"].dtype),
                sglang["seq_lens"].cumsum(dim=0),
            ]
        )
        report_diff("mini_cu_seqlens_k_vs_sglang_seq_cumsum", mini_cu_k, sglang_cu_k)


def print_backend_specific_fields(layer_id: int, mini: dict[str, Any], sglang: dict[str, Any]) -> None:
    print(f"=== Layer {layer_id} Backend-Specific Fields ===")
    mini_md = mini["attn_metadata"]
    sg_md = sglang["attn_metadata"]

    print("mini_attn_metadata:")
    for key in sorted(mini_md.keys()):
        value = mini_md[key]
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value!r}")

    print("sglang_attn_metadata:")
    for key in sorted(sg_md.keys()):
        value = sg_md[key]
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value!r}")


def compare_outputs(layer_id: int) -> None:
    print(f"=== Layer {layer_id} Backend Output ===")
    mini_out = load_pt(ATTN_DEBUG_DIR / f"l{layer_id}_after_attn_backend_attn_output.pt")
    sg_out = load_pt(ATTN_DEBUG_DIR / f"sglang_l{layer_id}_after_attn_backend_attn_output.pt")
    report_aligned_diff("attn_output", flatten_last_two(mini_out), flatten_last_two(sg_out))
    report_aligned_head_diff("attn_output", mini_out, sg_out, NUM_Q_HEADS)


def main() -> None:
    for layer_id in LAYERS:
        mini = load_pt(ATTN_DEBUG_DIR / f"l{layer_id}_mini_attn_backend_meta.pt")
        sglang = load_pt(ATTN_DEBUG_DIR / f"sglang_l{layer_id}_attn_backend_meta.pt")

        compare_common_fields(layer_id, mini, sglang)
        compare_batch_semantics(layer_id, mini, sglang)
        compare_visibility_semantics(layer_id, mini, sglang)
        print_backend_specific_fields(layer_id, mini, sglang)
        compare_outputs(layer_id)
        print()


if __name__ == "__main__":
    main()
