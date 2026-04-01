import torch


MINI_PATH = "tmp/l0_before_attn_hidden_states.pt"
SGLANG_PATH = "tmp/sglang_l0_before_attn_hidden_states.pt"


def main():
    mini = torch.load(MINI_PATH)
    sglang = torch.load(SGLANG_PATH)

    print(f"mini.shape=={mini.shape}, mini.dtype=={mini.dtype}, mini.device=={mini.device}")
    print(
        f"sglang.shape=={sglang.shape}, sglang.dtype=={sglang.dtype}, sglang.device=={sglang.device}"
    )

    if mini.shape != sglang.shape:
        raise ValueError(f"Shape mismatch: {mini.shape} vs {sglang.shape}")

    diff = (mini.float() - sglang.float()).abs()

    print(f"global_max_abs_diff=={diff.max().item():.6g}")
    print(f"global_mean_abs_diff=={diff.mean().item():.6g}")

    per_token_max = diff.amax(dim=-1)
    per_token_mean = diff.mean(dim=-1)
    print(f"per_token_max_abs_diff=={per_token_max}")
    print(f"per_token_mean_abs_diff=={per_token_mean}")

    print(f"mini_sum=={mini.sum(dim=-1)}")
    print(f"sglang_sum=={sglang.sum(dim=-1)}")
    print(f"sum_abs_diff=={(mini.sum(dim=-1).float() - sglang.sum(dim=-1).float()).abs()}")

    flat_diff = diff.flatten()
    k = min(20, flat_diff.numel())
    top_vals, top_idx = torch.topk(flat_diff, k=k)
    hidden_size = mini.shape[-1]
    print("top_diff_entries:")
    for rank, (val, idx) in enumerate(zip(top_vals.tolist(), top_idx.tolist()), start=1):
        token_idx = idx // hidden_size
        hidden_idx = idx % hidden_size
        mini_val = mini[token_idx, hidden_idx].item()
        sglang_val = sglang[token_idx, hidden_idx].item()
        print(
            f"{rank}: token={token_idx}, hidden={hidden_idx}, "
            f"abs_diff={val:.6g}, mini={mini_val}, sglang={sglang_val}"
        )


if __name__ == "__main__":
    main()
