import torch
from pathlib import Path


NUM_Q_HEADS = 10
NUM_KV_HEADS = 2
HEAD_DIM = 128
BASE_DIR = Path(__file__).resolve().parent


def load(name: str):
    path = BASE_DIR / name
    if not path.exists():
        path = BASE_DIR / "attn_debug" / name
    return torch.load(path, map_location="cpu")


def flatten_last_two(x: torch.Tensor) -> torch.Tensor:
    if x.ndim >= 3:
        return x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1])
    return x


def maybe_strip_leading_bos(mini: torch.Tensor, sglang: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if mini.ndim == 0 or sglang.ndim == 0:
        return mini, sglang, False
    if mini.shape[0] == sglang.shape[0] + 1:
        return mini[1:], sglang, True
    return mini, sglang, False


def report_diff(name: str, mini: torch.Tensor, sglang: torch.Tensor):
    mini = mini.float()
    sglang = sglang.float()
    if mini.shape != sglang.shape:
        print(f"{name}: shape mismatch {tuple(mini.shape)} vs {tuple(sglang.shape)}")
        return
    diff = (mini - sglang).abs()
    print(
        f"{name}: shape={tuple(mini.shape)}, "
        f"max_abs_diff={diff.max().item():.6g}, "
        f"mean_abs_diff={diff.mean().item():.6g}"
    )


def report_head_diff(name: str, mini: torch.Tensor, sglang: torch.Tensor, num_heads: int):
    mini = flatten_last_two(mini).float().view(-1, num_heads, HEAD_DIM)
    sglang = flatten_last_two(sglang).float().view(-1, num_heads, HEAD_DIM)
    diff = (mini - sglang).abs().amax(dim=(0, 2))
    print(f"{name}_head_diff=={diff}")


def compare_pair(
    label: str,
    mini_name: str,
    sglang_name: str,
    num_heads: int | None = None,
    flatten: bool = False,
):
    mini = load(mini_name)
    sglang = load(sglang_name)

    if flatten:
        mini = flatten_last_two(mini)
        sglang = flatten_last_two(sglang)

    mini, sglang, stripped = maybe_strip_leading_bos(mini, sglang)
    if stripped:
        print(f"{label}: stripped leading BOS row from mini before comparison")

    report_diff(label, mini, sglang)
    if num_heads is not None:
        report_head_diff(label, mini, sglang, num_heads)


def main():
    compare_pair(
        "l0_before_attn_hidden_states",
        "l0_before_attn_hidden_states.pt",
        "sglang_l0_before_attn_hidden_states.pt",
    )
    compare_pair(
        "l0_attn_after_qkv_proj",
        "l0_attn_after_qkv_proj.pt",
        "sglang_l0_attn_after_qkv_proj.pt",
    )
    compare_pair(
        "l0_after_rope_q",
        "l0_after_rope_q.pt",
        "sglang_l0_after_rope_q.pt",
        num_heads=NUM_Q_HEADS,
    )
    compare_pair(
        "l0_after_rope_k",
        "l0_after_rope_k.pt",
        "sglang_l0_after_rope_k.pt",
        num_heads=NUM_KV_HEADS,
    )
    compare_pair(
        "l0_after_rope_v",
        "l0_after_rope_v.pt",
        "sglang_l0_after_rope_v.pt",
        num_heads=NUM_KV_HEADS,
    )
    compare_pair(
        "l0_after_qknorm_q",
        "l0_after_qknorm_q.pt",
        "sglang_l0_after_qknorm_q.pt",
        num_heads=NUM_Q_HEADS,
    )
    compare_pair(
        "l0_after_qknorm_k",
        "l0_after_qknorm_k.pt",
        "sglang_l0_after_qknorm_k.pt",
        num_heads=NUM_KV_HEADS,
    )
    compare_pair(
        "l0_after_qknorm_v",
        "l0_after_qknorm_v.pt",
        "sglang_l0_after_qknorm_v.pt",
        num_heads=NUM_KV_HEADS,
    )
    compare_pair(
        "l0_before_attn_backend_q",
        "l0_before_attn_backend_q.pt",
        "sglang_l0_before_attn_backend_q.pt",
        num_heads=NUM_Q_HEADS,
        flatten=True,
    )
    compare_pair(
        "l0_before_attn_backend_k",
        "l0_before_attn_backend_k.pt",
        "sglang_l0_before_attn_backend_k.pt",
        num_heads=NUM_KV_HEADS,
    )
    compare_pair(
        "l0_before_attn_backend_v",
        "l0_before_attn_backend_v.pt",
        "sglang_l0_before_attn_backend_v.pt",
        num_heads=NUM_KV_HEADS,
    )
    compare_pair(
        "l0_after_attn_backend_attn_output",
        "l0_after_attn_backend_attn_output.pt",
        "sglang_l0_after_attn_backend_attn_output.pt",
        num_heads=NUM_Q_HEADS,
        flatten=True,
    )
    compare_pair(
        "l0_after_attn_hidden_states",
        "l0_after_attn_hidden_states.pt",
        "sglang_l0_after_attn_hidden_states.pt",
    )

    compare_pair(
        "l1_before_attn_hidden_states",
        "l1_before_attn_hidden_states.pt",
        "sglang_l1_before_attn_hidden_states.pt",
    )
    compare_pair(
        "l1_attn_after_qkv_proj",
        "l1_attn_after_qkv_proj.pt",
        "sglang_l1_attn_after_qkv_proj.pt",
    )
    compare_pair(
        "l1_after_rope_q",
        "l1_after_rope_q.pt",
        "sglang_l1_after_rope_q.pt",
        num_heads=NUM_Q_HEADS,
    )
    compare_pair(
        "l1_after_rope_k",
        "l1_after_rope_k.pt",
        "sglang_l1_after_rope_k.pt",
        num_heads=NUM_KV_HEADS,
    )
    compare_pair(
        "l1_after_rope_v",
        "l1_after_rope_v.pt",
        "sglang_l1_after_rope_v.pt",
        num_heads=NUM_KV_HEADS,
    )
    compare_pair(
        "l1_after_qknorm_q",
        "l1_after_qknorm_q.pt",
        "sglang_l1_after_qknorm_q.pt",
        num_heads=NUM_Q_HEADS,
    )
    compare_pair(
        "l1_after_qknorm_k",
        "l1_after_qknorm_k.pt",
        "sglang_l1_after_qknorm_k.pt",
        num_heads=NUM_KV_HEADS,
    )
    compare_pair(
        "l1_after_qknorm_v",
        "l1_after_qknorm_v.pt",
        "sglang_l1_after_qknorm_v.pt",
        num_heads=NUM_KV_HEADS,
    )
    compare_pair(
        "l1_before_attn_backend_q",
        "l1_before_attn_backend_q.pt",
        "sglang_l1_before_attn_backend_q.pt",
        num_heads=NUM_Q_HEADS,
        flatten=True,
    )
    compare_pair(
        "l1_before_attn_backend_k",
        "l1_before_attn_backend_k.pt",
        "sglang_l1_before_attn_backend_k.pt",
        num_heads=NUM_KV_HEADS,
    )
    compare_pair(
        "l1_before_attn_backend_v",
        "l1_before_attn_backend_v.pt",
        "sglang_l1_before_attn_backend_v.pt",
        num_heads=NUM_KV_HEADS,
    )
    compare_pair(
        "l1_attn_after_attention",
        "l1_attn_after_attention.pt",
        "sglang_l1_attn_after_attention.pt",
        flatten=True,
    )
    compare_pair(
        "l1_after_attn_hidden_states",
        "l1_after_attn_hidden_states.pt",
        "sglang_l1_after_attn_hidden_states.pt",
    )


if __name__ == "__main__":
    main()
