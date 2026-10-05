"""Character-level decoder-only Transformer language model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.0
    block_size: int = 256

    def __post_init__(self) -> None:
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if self.d_model < 1 or self.n_heads < 1 or self.n_layers < 1:
            raise ValueError("d_model, n_heads, and n_layers must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.d_ff < 1 or self.block_size < 1:
            raise ValueError("d_ff and block_size must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence_length = inputs.size(1)
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=inputs.device,
            ),
            diagonal=1,
        )
        attended, _ = self.attention(
            inputs,
            inputs,
            inputs,
            attn_mask=causal_mask,
            need_weights=False,
        )
        return self.residual_dropout(attended)


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.attention(self.attention_norm(inputs))
        return inputs + self.mlp(self.mlp_norm(inputs))


class GPTLanguageModel(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*(TransformerBlock(config) for _ in range(config.n_layers)))
        self.final_norm = nn.LayerNorm(config.d_model)
        self.language_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, sequence_length = token_ids.shape
        if sequence_length > self.config.block_size:
            raise ValueError("input sequence exceeds configured block_size")

        positions = torch.arange(sequence_length, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        hidden = self.blocks(self.embedding_dropout(hidden))
        logits = self.language_head(self.final_norm(hidden))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch_size * sequence_length, self.config.vocab_size),
                targets.reshape(batch_size * sequence_length),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.config.block_size :]
            logits, _ = self(context)
            probabilities = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)
        return token_ids


def config_to_dict(config: GPTConfig) -> dict[str, int | float]:
    """Return a serialization-friendly model configuration."""
    return asdict(config)