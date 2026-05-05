"""Language model modules for the multidomain benchmark."""

from __future__ import annotations

import torch
import torch.nn as nn


class BeliefConditionedCharTransformer(nn.Module):
    """Small causal character Transformer with optional belief-prefix tokens."""

    def __init__(
        self,
        vocab_size: int,
        *,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_sequence_length: int,
        belief_dim: int,
        prefix_tokens: int,
        use_belief_prefix: bool,
        handoff_mode: str | None = None,
    ):
        super().__init__()
        mode = handoff_mode or ("prefix" if use_belief_prefix else "none")
        self.handoff_mode = str(mode)
        self.use_belief_prefix = self.handoff_mode == "prefix"
        self.use_belief_adapter = self.handoff_mode == "adapter"
        self.prefix_tokens = int(prefix_tokens if self.use_belief_prefix else 0)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(
            int(max_sequence_length) + self.prefix_tokens + 4,
            d_model,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.belief_projector = nn.Sequential(
            nn.Linear(belief_dim + 2, d_model),
            nn.Tanh(),
            nn.Linear(d_model, max(1, self.prefix_tokens) * d_model),
        )
        self.belief_adapter = nn.Sequential(
            nn.Linear(belief_dim + 2, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 2 * d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def belief_features(
        self,
        belief_vector: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        confidence: float = 1.0,
        uncertainty: float = 0.0,
    ) -> torch.Tensor:
        """Return one batch-aligned belief feature tensor."""
        belief_dim = self.belief_projector[0].in_features - 2
        if belief_vector is None:
            belief_vector = torch.zeros((batch_size, belief_dim), dtype=torch.float32, device=device)
        belief_vector = belief_vector.to(device=device, dtype=torch.float32)
        if belief_vector.ndim == 1:
            belief_vector = belief_vector.unsqueeze(0).expand(batch_size, -1)
        extras = torch.tensor(
            [float(confidence), float(uncertainty)],
            dtype=torch.float32,
            device=device,
        ).view(1, 2).expand(batch_size, -1)
        return torch.cat([belief_vector, extras], dim=-1)

    def prefix_from_belief(
        self,
        belief_vector: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        confidence: float = 1.0,
        uncertainty: float = 0.0,
    ) -> torch.Tensor | None:
        """Project one belief into continuous prefix tokens."""
        if not self.use_belief_prefix:
            return None
        prefix = self.belief_projector(
            self.belief_features(
                belief_vector,
                batch_size=batch_size,
                device=device,
                confidence=confidence,
                uncertainty=uncertainty,
            )
        )
        return prefix.view(batch_size, self.prefix_tokens, -1)

    def apply_belief_adapter(
        self,
        hidden: torch.Tensor,
        belief_vector: torch.Tensor | None,
        *,
        confidence: float,
        uncertainty: float,
    ) -> torch.Tensor:
        """Apply a small FiLM-style residual adapter from the belief."""
        if not self.use_belief_adapter:
            return hidden
        features = self.belief_features(
            belief_vector,
            batch_size=hidden.shape[0],
            device=hidden.device,
            confidence=confidence,
            uncertainty=uncertainty,
        )
        gamma_beta = self.belief_adapter(features).unsqueeze(1)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)
        return hidden * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * torch.tanh(beta)

    def forward(
        self,
        tokens: torch.Tensor,
        belief_vector: torch.Tensor | None = None,
        *,
        confidence: float = 1.0,
        uncertainty: float = 0.0,
    ) -> torch.Tensor:
        batch_size, token_count = tokens.shape
        embeddings = self.token_embedding(tokens)
        prefix = self.prefix_from_belief(
            belief_vector,
            batch_size=batch_size,
            device=tokens.device,
            confidence=confidence,
            uncertainty=uncertainty,
        )
        sequence = torch.cat([prefix, embeddings], dim=1) if prefix is not None else embeddings
        positions = torch.arange(sequence.shape[1], device=tokens.device)
        sequence = sequence + self.position_embedding(positions).unsqueeze(0)
        mask = torch.triu(
            torch.ones(
                sequence.shape[1],
                sequence.shape[1],
                dtype=torch.bool,
                device=tokens.device,
            ),
            diagonal=1,
        )
        hidden = self.transformer(sequence, mask=mask)
        if prefix is not None:
            hidden = hidden[:, self.prefix_tokens :, :]
        hidden = self.apply_belief_adapter(
            hidden,
            belief_vector,
            confidence=confidence,
            uncertainty=uncertainty,
        )
        return self.head(self.norm(hidden[:, :token_count, :]))
