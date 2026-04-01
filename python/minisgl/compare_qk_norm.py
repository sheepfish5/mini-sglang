from __future__ import annotations

from pathlib import Path

import torch


TMP_DIR = Path("tmp")

MINI_BEFORE_QKV = TMP_DIR / "l0_before_qknorm_qkv.pt"
SGLANG_BEFORE_QKV = TMP_DIR / "sglang_l0_before_qknorm_qkv.pt"

MINI_AFTER_Q = TMP_DIR / "l0_after_qknorm_q.pt"
MINI_AFTER_K = TMP_DIR / "l0_after_qknorm_k.pt"
MINI_AFTER_V = TMP_DIR / "l0_after_qknorm_v.pt"

SGLANG_AFTER_Q = TMP_DIR / "sglang_l0_after_qknorm_q.pt"
SGLANG_AFTER_K = TMP_DIR / "sglang_l0_after_qknorm_k.pt"
SGLANG_AFTER_V = TMP_DIR / "sglang_l0_after_qknorm_v.pt"

MINI_WEIGHT = TMP_DIR / "l0_before_qknorm_qk_norm_weight.pt"
SGLANG_WEIGHT = TMP_DIR / "sglang_l0_before_qknorm_qk_norm_weight.pt"

NUM_Q_HEADS = 10
NUM_KV_HEADS = 2
HEAD_DIM = 128
Q_DIM = NUM_Q_HEADS * HEAD_DIM
KV_DIM = NUM_KV_HEADS * HEAD_DIM


def load_tensor(path: Path) -> torch.Tensor:
    x = torch.load(path)
    if isinstance(x, torch.Tensor):
        return x
    if hasattr(x, "data") and isinstance(x.data, torch.Tensor):
        return x.data
    raise TypeError(f"{path}: unsupported type {type(x)}")


def report_diff(name: str, mini: torch.Tensor, sglang: torch.Tensor):
    mini_f = mini.float()
    sglang_f = sglang.float()
    if mini.shape != sglang.shape:
        print(f"{name}: shape mismatch {tuple(mini.shape)} vs {tuple(sglang.shape)}")
        return
    diff = (mini_f - sglang_f).abs()
    print(
        f"{name}: shape={tuple(mini.shape)}, "
        f"max_abs_diff={diff.max().item():.6g}, "
        f"mean_abs_diff={diff.mean().item():.6g}"
    )


def report_head_diff(name: str, mini: torch.Tensor, sglang: torch.Tensor, num_heads: int):
    mini_h = mini.float().view(-1, num_heads, HEAD_DIM)
    sglang_h = sglang.float().view(-1, num_heads, HEAD_DIM)
    diff = (mini_h - sglang_h).abs().amax(dim=(0, 2))
    print(f"{name}_head_diff=={diff}")


def report_top_diff(name: str, mini: torch.Tensor, sglang: torch.Tensor, limit: int = 20):
    diff = (mini.float() - sglang.float()).abs().flatten()
    k = min(limit, diff.numel())
    vals, idxs = torch.topk(diff, k=k)
    last_dim = mini.shape[-1] if mini.ndim > 1 else 1
    print(f"top_{name}_diff:")
    for rank, (val, idx) in enumerate(zip(vals.tolist(), idxs.tolist()), start=1):
        row = idx // last_dim if mini.ndim > 1 else 0
        col = idx % last_dim if mini.ndim > 1 else idx
        mini_val = mini.flatten()[idx].item()
        sglang_val = sglang.flatten()[idx].item()
        print(
            f"{rank}: row={row}, col={col}, abs_diff={val:.6g}, "
            f"mini={mini_val}, sglang={sglang_val}"
        )


def main():
    mini_before_qkv = load_tensor(MINI_BEFORE_QKV)
    sglang_before_qkv = load_tensor(SGLANG_BEFORE_QKV)

    mini_after_q = load_tensor(MINI_AFTER_Q)
    mini_after_k = load_tensor(MINI_AFTER_K)
    mini_after_v = load_tensor(MINI_AFTER_V)
    sglang_after_q = load_tensor(SGLANG_AFTER_Q)
    sglang_after_k = load_tensor(SGLANG_AFTER_K)
    sglang_after_v = load_tensor(SGLANG_AFTER_V)

    mini_weight = load_tensor(MINI_WEIGHT)
    sglang_weight = load_tensor(SGLANG_WEIGHT)

    print(f"mini_before_qkv.shape=={mini_before_qkv.shape}, dtype=={mini_before_qkv.dtype}")
    print(f"sglang_before_qkv.shape=={sglang_before_qkv.shape}, dtype=={sglang_before_qkv.dtype}")
    print(f"mini_after_q.shape=={mini_after_q.shape}, dtype=={mini_after_q.dtype}")
    print(f"mini_after_k.shape=={mini_after_k.shape}, dtype=={mini_after_k.dtype}")
    print(f"mini_after_v.shape=={mini_after_v.shape}, dtype=={mini_after_v.dtype}")
    print(f"sglang_after_q.shape=={sglang_after_q.shape}, dtype=={sglang_after_q.dtype}")
    print(f"sglang_after_k.shape=={sglang_after_k.shape}, dtype=={sglang_after_k.dtype}")
    print(f"sglang_after_v.shape=={sglang_after_v.shape}, dtype=={sglang_after_v.dtype}")
    print(f"mini_weight.shape=={mini_weight.shape}, dtype=={mini_weight.dtype}")
    print(f"sglang_weight.shape=={sglang_weight.shape}, dtype=={sglang_weight.dtype}")

    report_diff("before_qknorm_qkv", mini_before_qkv, sglang_before_qkv)
    report_diff("qk_norm_weight", mini_weight, sglang_weight)

    mini_before_q, mini_before_k, mini_before_v = mini_before_qkv.split([Q_DIM, KV_DIM, KV_DIM], dim=-1)
    sglang_before_q, sglang_before_k, sglang_before_v = sglang_before_qkv.split([Q_DIM, KV_DIM, KV_DIM], dim=-1)

    report_diff("before_qknorm_q", mini_before_q, sglang_before_q)
    report_diff("before_qknorm_k", mini_before_k, sglang_before_k)
    report_diff("before_qknorm_v", mini_before_v, sglang_before_v)
    report_diff("after_qknorm_q", mini_after_q, sglang_after_q)
    report_diff("after_qknorm_k", mini_after_k, sglang_after_k)
    report_diff("after_qknorm_v", mini_after_v, sglang_after_v)

    report_head_diff("before_qknorm_q", mini_before_q, sglang_before_q, NUM_Q_HEADS)
    report_head_diff("before_qknorm_k", mini_before_k, sglang_before_k, NUM_KV_HEADS)
    report_head_diff("after_qknorm_q", mini_after_q, sglang_after_q, NUM_Q_HEADS)
    report_head_diff("after_qknorm_k", mini_after_k, sglang_after_k, NUM_KV_HEADS)
    report_head_diff("after_qknorm_v", mini_after_v, sglang_after_v, NUM_KV_HEADS)

    mini_q_delta = mini_after_q.float() - mini_before_q.float()
    sglang_q_delta = sglang_after_q.float() - sglang_before_q.float()
    mini_k_delta = mini_after_k.float() - mini_before_k.float()
    sglang_k_delta = sglang_after_k.float() - sglang_before_k.float()
    mini_v_delta = mini_after_v.float() - mini_before_v.float()
    sglang_v_delta = sglang_after_v.float() - sglang_before_v.float()

    report_diff("qknorm_delta_q", mini_q_delta, sglang_q_delta)
    report_diff("qknorm_delta_k", mini_k_delta, sglang_k_delta)
    report_diff("qknorm_delta_v", mini_v_delta, sglang_v_delta)
    report_head_diff("qknorm_delta_q", mini_q_delta, sglang_q_delta, NUM_Q_HEADS)
    report_head_diff("qknorm_delta_k", mini_k_delta, sglang_k_delta, NUM_KV_HEADS)

    report_top_diff("after_qknorm_q", mini_after_q, sglang_after_q)
    report_top_diff("after_qknorm_k", mini_after_k, sglang_after_k)
    report_top_diff("after_qknorm_v", mini_after_v, sglang_after_v)
    report_top_diff("qk_norm_weight", mini_weight, sglang_weight)


if __name__ == "__main__":
    main()
