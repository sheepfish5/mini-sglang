from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP as Qwen3MLP
from .utils import RopeAttn as Qwen3Attn

if TYPE_CHECKING:
    from .config import ModelConfig


class Qwen3DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = Qwen3Attn(config, layer_id, has_qk_norm=True)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class Qwen3Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen3DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids_list = [151644, 872, 198, 9707, 0, 151645, 198, 151644, 77091, 198]
        input_ids = torch.tensor(input_ids_list, dtype=input_ids.dtype, device=input_ids.device)
        x = self.embed_tokens.forward(input_ids)
        torch.save(x, "tmp/embeddings.pt")
        print(f"[Qwen3Model.forward] embeddings.shape=={x.shape}")
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        
        return self.norm.forward(x, residual)[0]


class Qwen3ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        print(f"[Qwen3ForCausalLM.forward] model_output.shape=={output.shape}")
        torch.save(output, "tmp/model_output.pt")
        logits = self.lm_head.forward(output)
        print(f"[Qwen3ForCausalLM.forward] logits.shape=={logits.shape}")
        torch.save(logits, "tmp/logits.pt")
        import os
        print("[Qwen3ForCausalLM.forward] os._exit(1)")
        os._exit(1)
        return logits


__all__ = ["Qwen3ForCausalLM"]
