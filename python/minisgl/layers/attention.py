from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import StateLessOP
from .rotary import get_rope

if TYPE_CHECKING:
    from minisgl.layers import RMSNorm
    from minisgl.models import RotaryConfig


class AttentionLayer(StateLessOP):
    def __init__(
        self,
        layer_id: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_config: RotaryConfig,
        q_norm: RMSNorm | None = None,
        k_norm: RMSNorm | None = None,
        has_rope: bool = True,
        attn_temperature_tuning: bool = False,
        floor_scale=8192,
        attn_scale=0.1,
    ):
        assert num_qo_heads % num_kv_heads == 0
        self.layer_id = layer_id
        self.head_dim = head_dim
        tp_size = get_tp_info().size
        self.num_qo_heads = div_even(num_qo_heads, tp_size)
        self.num_kv_heads = div_even(num_kv_heads, tp_size)
        self.qo_attn_dim = self.num_qo_heads * head_dim
        self.kv_attn_dim = self.num_kv_heads * head_dim
        self.has_rope = has_rope
        if has_rope:
            self.rotary = get_rope(
                head_dim=head_dim,
                rotary_dim=rotary_config.rotary_dim,
                max_position=rotary_config.max_position,
                base=rotary_config.base,
                rope_scaling=tuple(rotary_config.scaling.items()) if rotary_config.scaling else None,
            )
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.attn_temperature_tuning = attn_temperature_tuning
        self.floor_scale = floor_scale
        self.attn_scale = attn_scale

        tp_info = get_tp_info()
        self.attn_tp_rank = tp_info.rank

    def _get_attn_scale(self, positions: torch.Tensor) -> torch.Tensor:
        floor = torch.floor((positions + 1.0) / self.floor_scale)
        attn_scale = torch.log(floor + 1.0) * self.attn_scale + 1.0
        return attn_scale.unsqueeze(-1).unsqueeze(-1)

    def _mul_attn_scale(self, positions, q: torch.Tensor) -> torch.Tensor:
        attn_scale = self._get_attn_scale(positions)
        return (q * attn_scale).to(q.dtype)

    def forward(self, qkv: torch.Tensor, rope_first: bool = False, qk_norm_combined: bool = False, qk_norm: RMSNorm | None = None) -> torch.Tensor:
        ctx = get_global_ctx()
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)

        if q.shape[0] == 13:
            self.debug_mode = True
        debug_ids = [0, 1]

        if rope_first and self.has_rope:
            if self.layer_id == 0:
                ctx.batch.positions = ctx.batch.positions + 1
            if self.layer_id in debug_ids and self.attn_tp_rank == 0 and hasattr(self, "debug_mode") and self.debug_mode:
                q_sum = torch.sum(q.view(13, -1), dim=-1)
                k_sum = torch.sum(k.view(13, -1), dim=-1)
                print(f"[AttentionLayer.forward] [{self.layer_id}] before_rope_q_sum=={q_sum}") # (13,)
                print(f"[AttentionLayer.forward] [{self.layer_id}] before_rope_k_sum=={k_sum}") # (13,)
                print(f"[AttentionLayer.forward] [{self.layer_id}] positions.shape=={ctx.batch.positions.shape}, positions=={ctx.batch.positions}") # (13,)
                print(f"[AttentionLayer.forward] [{self.layer_id}] q.shape=={q.shape}, q.stride=={q.stride()}") # (13, 1280)
                print(f"[AttentionLayer.forward] [{self.layer_id}] k.shape=={k.shape}, k.stride=={k.stride()}") # (13, 1280)
            self.rotary.forward(ctx.batch.positions, q, k)
            # print("执行 rope first")
        
        if self.layer_id in debug_ids and self.attn_tp_rank == 0 and hasattr(self, "debug_mode") and self.debug_mode:
            print(f"[AttentionLayer.forward] [{self.layer_id}] after_rope.shape=={q.shape}") # (13, 1280)
            print(f"[AttentionLayer.forward] [{self.layer_id}] self.rotary._cos_sin_cache.shape=={self.rotary._cos_sin_cache.shape}") # (13, 1280)
            torch.save(q, f"/root/autodl-tmp/mini-sglang/tmp/l{self.layer_id}_after_rope_q.pt") 
            torch.save(k, f"/root/autodl-tmp/mini-sglang/tmp/l{self.layer_id}_after_rope_k.pt")
            torch.save(v, f"/root/autodl-tmp/mini-sglang/tmp/l{self.layer_id}_after_rope_v.pt")
            torch.save(self.rotary._cos_sin_cache[:1024, :].clone(), f"/root/autodl-tmp/mini-sglang/tmp/l{self.layer_id}_after_rope_cos_sin_cache.pt")
            positions = ctx.batch.positions
            print(f"[AttentionLayer.forward] [{self.layer_id}] positions.shape=={positions.shape}, positions=={positions}")

        if qk_norm_combined and qk_norm is not None:
            if self.layer_id in debug_ids and self.attn_tp_rank == 0 and hasattr(self, "debug_mode") and self.debug_mode:
                print("执行 qk_norm")
                print(f"[AttentionLayer.forward] [{self.layer_id}] before_qk_norm ")
                torch.save(qkv, f"/root/autodl-tmp/mini-sglang/tmp/sglang_l{self.layer_id}_before_qknorm_qkv.pt")
                torch.save(qk_norm.weight, f"/root/autodl-tmp/mini-sglang/tmp/sglang_l{self.layer_id}_before_qknorm_qk_norm_weight.pt")

            # qk_norm.forward_inplace(torch.stack([q.view(-1, self.num_qo_heads, self.head_dim), k.view(-1, self.num_kv_heads, self.head_dim)]))
            qk, _ = qkv.split([self.qo_attn_dim + self.kv_attn_dim, self.kv_attn_dim], dim=-1)
            qk_norm.forward_inplace(qk.view(-1, self.num_qo_heads + self.num_kv_heads, self.head_dim))
            del qk
            
            if self.layer_id in debug_ids and self.attn_tp_rank == 0 and hasattr(self, "debug_mode") and self.debug_mode:
                print(f"[AttentionLayer.forward] [{self.layer_id}] after_qknorm")
                torch.save(qkv, f"/root/autodl-tmp/mini-sglang/tmp/sglang_l{self.layer_id}_after_qknorm_qkv.pt")    
        else:
            if self.q_norm is not None:
                self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
            if self.k_norm is not None:
                self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        if not rope_first and self.has_rope:
            q, k = self.rotary.forward(ctx.batch.positions, q, k)
        
        if self.layer_id in debug_ids and self.attn_tp_rank == 0 and hasattr(self, "debug_mode") and self.debug_mode:
            print(f"[AttentionLayer.forward] [{self.layer_id}] after_qknorm.shape=={q.shape}") # (13, 1280)
            torch.save(q, f"/root/autodl-tmp/mini-sglang/tmp/l{self.layer_id}_after_qknorm_q.pt")
            torch.save(k, f"/root/autodl-tmp/mini-sglang/tmp/l{self.layer_id}_after_qknorm_k.pt")
            torch.save(v, f"/root/autodl-tmp/mini-sglang/tmp/l{self.layer_id}_after_qknorm_v.pt")

        # if self.q_norm is not None:
        #     self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        # if self.k_norm is not None:
        #     self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        # if self.has_rope:
        #     q, k = self.rotary.forward(ctx.batch.positions, q, k)
        q = q.view(-1, self.num_qo_heads, self.head_dim)

        # We are applying temperature tuning (https://arxiv.org/abs/2501.19399) to NoPE layers, where
        # the inference-time temperature tuning function is customized to not affect short context
        # while working at very long context
        # https://arxiv.org/abs/2501.19399
        if self.attn_temperature_tuning and not self.has_rope:
            q = self._mul_attn_scale(positions=ctx.batch.positions, q=q)

        if self.layer_id in debug_ids and self.attn_tp_rank == 0 and hasattr(self, "debug_mode") and self.debug_mode:
            print(f"[AttentionLayer.forward] [{self.layer_id}] before_attn_backend.shape=={q.shape}")
            torch.save(q, f"/root/autodl-tmp/mini-sglang/tmp/l{self.layer_id}_before_attn_backend_q.pt")
            torch.save(k, f"/root/autodl-tmp/mini-sglang/tmp/l{self.layer_id}_before_attn_backend_k.pt")
            torch.save(v, f"/root/autodl-tmp/mini-sglang/tmp/l{self.layer_id}_before_attn_backend_v.pt")


        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        return o.view(-1, self.qo_attn_dim)
