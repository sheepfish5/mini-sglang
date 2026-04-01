# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.6.6.post1/vllm/model_executor/layers/rotary_embedding.py

"""Rotary Positional Embeddings."""
import itertools
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import triton
import triton.language as tl

from sglang.srt.custom_op import CustomOp
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    get_compiler_backend,
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
    is_xpu,
)

_is_cuda = is_cuda()
_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_npu = is_npu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_is_xpu = is_xpu()

if _is_cuda:
    from sgl_kernel import FusedSetKVBufferArg, apply_rope_with_cos_sin_cache_inplace
else:
    FusedSetKVBufferArg = None

if _use_aiter:
    from aiter.rotary_embedding import get_rope as aiter_get_rope

if is_npu():
    import torch_npu

    NPU_ROTARY_MUL_MAX_NUM_HEADS = 1000
    NPU_ROTARY_MUL_MAX_HEAD_SIZE = 896


def _rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


class RotaryEmbedding(CustomOp):
    """Original rotary positional embedding."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        cache = self._compute_cos_sin_cache()
        # NOTE(ByronHsu): cache needs to be in FP32 for numerical stability
        if not _is_cuda:
            cache = cache.to(dtype)

        if (
            (not (_is_cuda or _is_npu) or self.head_size not in [64, 128, 256, 512])
            and not (_is_cpu and _is_cpu_amx_available)
            and not (_is_xpu)
        ):
            if _is_cuda:
                from sgl_kernel import rotary_embedding
            else:
                from vllm._custom_ops import rotary_embedding

            self.use_fallback_kernel = True
            self.fallback_rotary_embedding = rotary_embedding
        else:
            self.use_fallback_kernel = False

        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

        self._apply_rotary_emb_wrapped = _apply_rotary_emb

        if get_global_server_args().rl_on_policy_target is not None:
            self._forward_method = self.forward_native
            self._apply_rotary_emb_wrapped = torch.compile(dynamic=True)(
                self._apply_rotary_emb_wrapped
            )
        self.position_cos, self.position_sin = None, None

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        init_device = (
            "cpu" if get_global_server_args().rl_on_policy_target is not None else None
        )
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(
                    0, self.rotary_dim, 2, dtype=torch.float, device=init_device
                )
                / self.rotary_dim
            )
        )
        if get_global_server_args().rl_on_policy_target is not None:
            inv_freq = inv_freq.cuda()
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def _ensure_cos_sin_cache_length(self, needed_max_pos: int):
        """Ensure cos_sin_cache length > needed_max_pos."""
        from sglang.srt.environ import envs

        cur_len = int(self.cos_sin_cache.shape[0])
        if needed_max_pos < cur_len:
            return

        # Align to reduce realloc frequency
        align = envs.SGLANG_ROPE_CACHE_ALIGN.value
        new_len = ((needed_max_pos + align) // align) * align
        device = self.cos_sin_cache.device
        dtype = self.cos_sin_cache.dtype

        # Compute inv_freq on same device
        inv_freq = self._compute_inv_freq(self.base).to(device=device)

        # Incremental computation for new positions only
        start = cur_len
        t_new = torch.arange(start, new_len, dtype=inv_freq.dtype, device=device)
        if t_new.numel() == 0:
            return

        freqs_new = torch.einsum("i,j->ij", t_new, inv_freq)
        cos_new = freqs_new.cos()
        sin_new = freqs_new.sin()
        new_rows = torch.cat((cos_new, sin_new), dim=-1).to(dtype=dtype)

        # Update cache with new rows
        self.cos_sin_cache = torch.cat((self.cos_sin_cache, new_rows), dim=0).to(
            device=device, dtype=dtype
        )

    def get_cos_sin_with_position(self, positions):
        cos_sin = self.cos_sin_cache.index_select(0, positions.flatten())
        last_dim = cos_sin.size()[-1]
        cos, sin = (
            cos_sin.reshape(-1, 2, last_dim // 2).repeat(1, 1, 2).chunk(2, dim=-2)
        )
        # BSNH
        self.position_cos, self.position_sin = (
            cos.view(-1, 1, 1, last_dim).contiguous(),
            sin.view(-1, 1, 1, last_dim).contiguous(),
        )

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        fused_set_kv_buffer_arg: Optional[FusedSetKVBufferArg] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A PyTorch-native implementation of forward()."""
        assert (
            fused_set_kv_buffer_arg is None
        ), "fused_set_kv_buffer_arg is not supported for native implementation"

        if offsets is not None:
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = self._apply_rotary_emb_wrapped(
            query_rot, cos, sin, self.is_neox_style
        )
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = self._apply_rotary_emb_wrapped(key_rot, cos, sin, self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    def forward_npu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        fused_set_kv_buffer_arg: Optional[FusedSetKVBufferArg] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A PyTorch-npu implementation of forward()."""
        assert (
            fused_set_kv_buffer_arg is None
        ), "fused_set_kv_buffer_arg is not supported for npu implementation"

        if get_bool_env_var("SGLANG_ENABLE_TORCH_COMPILE"):
            return self.forward_native(
                positions, query, key, offsets, fused_set_kv_buffer_arg
            )
        else:
            rotary_mode = "half"
            if self.is_neox_style:
                rotary_mode = "half"
            else:
                rotary_mode = "interleave"
            mrope_section = [0, 0, 0]
            query_out, key_out = torch_npu.npu_mrope(
                positions,
                query,
                key,
                self.cos_sin_cache,
                self.head_size,
                mrope_section=mrope_section,
                rotary_mode=rotary_mode,
            )
            return query_out, key_out

    def forward_cpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        fused_set_kv_buffer_arg: Optional[FusedSetKVBufferArg] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (
            fused_set_kv_buffer_arg is None
        ), "fused_set_kv_buffer_arg is not supported for cpu implementation"

        positions = torch.add(positions, offsets) if offsets is not None else positions
        if _is_cpu_amx_available:
            return torch.ops.sgl_kernel.rotary_embedding_cpu(
                positions,
                query,
                key,
                self.head_size,
                self.cos_sin_cache,
                self.is_neox_style,
            )
        else:
            return self.forward_native(
                positions, query, key, offsets, fused_set_kv_buffer_arg
            )

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        fused_set_kv_buffer_arg: Optional[FusedSetKVBufferArg] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_fallback_kernel:
            apply_rope_with_cos_sin_cache_inplace(
                positions=positions,
                query=query,
                key=key,
                head_size=self.head_size,
                cos_sin_cache=self.cos_sin_cache,
                is_neox=self.is_neox_style,
                # Compatible with old sgl-kernel
                **(
                    dict(fused_set_kv_buffer_arg=fused_set_kv_buffer_arg)
                    if fused_set_kv_buffer_arg is not None
                    else {}
                ),
            )
        else:
            assert (
                fused_set_kv_buffer_arg is None
            ), "save kv cache is not supported for fallback_rotary_embedding."
            self.cos_sin_cache = self.cos_sin_cache.to(query.device, dtype=query.dtype)
            self.fallback_rotary_embedding(
                positions,
                query,
                key,
                self.head_size,
                self.cos_sin_cache,
                self.is_neox_style,
            )
        return query, key

    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        s += f", max_position_embeddings={self.max_position_embeddings}"
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"
        return s

    def forward_xpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        fused_set_kv_buffer_arg: Optional[FusedSetKVBufferArg] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (
            fused_set_kv_buffer_arg is None
        ), "fused_set_kv_buffer_arg is not supported for xpu implementation"
        positions = torch.add(positions, offsets) if offsets is not None else positions
        return torch.ops.sgl_kernel.rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
            self.is_neox_style,
        )



class Llama3RotaryEmbedding(RotaryEmbedding):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
        scaling_factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        orig_max_position: int,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.orig_max_position = orig_max_position
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        inv_freqs = super()._compute_inv_freq(base)
        low_freq_wavelen = self.orig_max_position / self.low_freq_factor
        high_freq_wavelen = self.orig_max_position / self.high_freq_factor

        wave_len = 2 * math.pi / inv_freqs
        if self.low_freq_factor != self.high_freq_factor:
            smooth = (self.orig_max_position / wave_len - self.low_freq_factor) / (
                self.high_freq_factor - self.low_freq_factor
            )
        else:
            smooth = 0
        new_freqs = torch.where(
            wave_len < high_freq_wavelen,
            inv_freqs,
            torch.where(
                wave_len > low_freq_wavelen,
                inv_freqs / self.scaling_factor,
                (1 - smooth) * inv_freqs / self.scaling_factor + smooth * inv_freqs,
            ),
        )
        return new_freqs
    
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    is_neox_style: bool = True,
    rope_scaling: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
    partial_rotary_factor: float = 1.0,
    dual_chunk_attention_config: Optional[Dict[str, Any]] = None,
) -> RotaryEmbedding:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if rope_scaling is not None:
        # Transforms every value that is a list into a tuple for caching calls
        rope_scaling_tuple = {
            k: tuple(v) if isinstance(v, list) else v for k, v in rope_scaling.items()
        }
        rope_scaling_args = tuple(rope_scaling_tuple.items())
    else:
        rope_scaling_args = None

    if dual_chunk_attention_config is not None:
        dual_chunk_attention_tuple = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in dual_chunk_attention_config.items()
            if k != "sparse_attention_config"
        }
        dual_chunk_attention_args = tuple(dual_chunk_attention_tuple.items())
    else:
        dual_chunk_attention_args = None

    if partial_rotary_factor < 1.0:
        rotary_dim = int(rotary_dim * partial_rotary_factor)
    key = (
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        rope_scaling_args,
        dual_chunk_attention_args,
        dtype,
    )
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]

    if dual_chunk_attention_config is not None:
        extra_kwargs = {
            k: v
            for k, v in dual_chunk_attention_config.items()
            if k in ("chunk_size", "local_size")
        }
        rotary_emb = DualChunkRotaryEmbedding(
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            dtype,
            **extra_kwargs,
        )
    elif rope_scaling is None:
        rotary_emb = RotaryEmbedding(
            head_size, rotary_dim, max_position, base, is_neox_style, dtype
        )
    else:
        if "rope_type" in rope_scaling:
            scaling_type = rope_scaling["rope_type"]
        elif "type" in rope_scaling:
            scaling_type = rope_scaling["type"]
        else:
            raise ValueError("Unknown RoPE scaling type")

        if scaling_type == "llama3":
            scaling_factor = rope_scaling["factor"]
            low_freq_factor = rope_scaling["low_freq_factor"]
            high_freq_factor = rope_scaling["high_freq_factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            rotary_emb = Llama3RotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                dtype,
                scaling_factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position,
            )
        elif scaling_type == "default":
            if "mrope_section" in rope_scaling:
                rotary_emb = MRotaryEmbedding(
                    head_size,
                    rotary_dim,
                    max_position,
                    base,
                    is_neox_style,
                    dtype,
                    mrope_section=rope_scaling["mrope_section"],
                    mrope_interleaved=rope_scaling.get("mrope_interleaved", False),
                )
            else:
                rotary_emb = RotaryEmbedding(
                    head_size,
                    rotary_dim,
                    max_position,
                    base,
                    is_neox_style,
                    dtype,
                )
        elif scaling_type == "linear":
            scaling_factor = rope_scaling["factor"]
            rotary_emb = LinearScalingRotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                scaling_factor,
                dtype,
            )
        elif scaling_type == "dynamic":
            scaling_factor = rope_scaling["factor"]
            if "alpha" in rope_scaling:
                rotary_emb = DynamicNTKAlphaRotaryEmbedding(
                    head_size,
                    rotary_dim,
                    max_position,
                    base,
                    is_neox_style,
                    rope_scaling["alpha"],
                    dtype,
                )
            else:
                rotary_emb = DynamicNTKScalingRotaryEmbedding(
                    head_size,
                    rotary_dim,
                    max_position,
                    base,
                    is_neox_style,
                    scaling_factor,
                    dtype,
                )
        elif scaling_type == "yarn":
            scaling_factor = rope_scaling["factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k
                in ("extrapolation_factor", "attn_factor", "beta_fast", "beta_slow")
            }
            rotary_emb = YaRNScalingRotaryEmbedding(
                head_size,
                rotary_dim,
                original_max_position,
                base,
                is_neox_style,
                scaling_factor,
                dtype,
                **extra_kwargs,
            )
        elif scaling_type == "deepseek_yarn":
            scaling_factor = rope_scaling["factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            # assert max_position == original_max_position * scaling_factor
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k
                in (
                    "extrapolation_factor",
                    "attn_factor",
                    "beta_fast",
                    "beta_slow",
                    "mscale",
                    "mscale_all_dim",
                )
            }
            rotary_emb = DeepseekScalingRotaryEmbedding(
                head_size,
                rotary_dim,
                original_max_position,
                base,
                is_neox_style,
                scaling_factor,
                dtype,
                **extra_kwargs,
            )
        elif scaling_type == "longrope":
            short_factor = rope_scaling["short_factor"]
            long_factor = rope_scaling["long_factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k in ("short_mscale", "long_mscale")
            }
            rotary_emb = Phi3LongRoPEScaledRotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                original_max_position,
                base,
                is_neox_style,
                dtype,
                short_factor,
                long_factor,
                **extra_kwargs,
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb