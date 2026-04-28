import torch
from torch import nn

from .token_schema import (
    TOKEN_COUNT,
    NUM_TOKEN_TYPES,
    NUM_TOKEN_SIDES,
    NUM_TOKEN_SLOTS,
    token_type_tensor,
    token_side_tensor,
    token_slot_tensor,
)


class MatchTokenEmbedding(nn.Module):
    """Build token embeddings as additive value/type/side/slot components."""

    def __init__(self, d_model: int):
        super(MatchTokenEmbedding, self).__init__()
        self.value_embedding = nn.Linear(1, d_model)
        self.type_embedding = nn.Embedding(NUM_TOKEN_TYPES, d_model)
        self.side_embedding = nn.Embedding(NUM_TOKEN_SIDES, d_model)
        self.slot_embedding = nn.Embedding(NUM_TOKEN_SLOTS, d_model)

        # Fixed schema vectors registered as buffers for automatic device placement.
        self.register_buffer("token_type_ids", token_type_tensor())
        self.register_buffer("token_side_ids", token_side_tensor())
        self.register_buffer("token_slot_ids", token_slot_tensor())

    def forward(self, token_values: torch.Tensor) -> torch.Tensor:
        """Return batch-first token embeddings [batch, seq_len, d_model]."""
        if token_values.dim() != 2:
            raise ValueError(f"token_values must be 2D [batch, seq_len], got {token_values.shape}")
        if token_values.size(1) != TOKEN_COUNT:
            raise ValueError(
                f"token_values seq_len must be {TOKEN_COUNT}, got {token_values.size(1)}"
            )

        batch_size, seq_len = token_values.shape
        value_embedding = self.value_embedding(token_values.unsqueeze(-1).float())

        type_ids = self.token_type_ids.unsqueeze(0).expand(batch_size, seq_len)
        side_ids = self.token_side_ids.unsqueeze(0).expand(batch_size, seq_len)
        slot_ids = self.token_slot_ids.unsqueeze(0).expand(batch_size, seq_len)

        return (
            value_embedding
            + self.type_embedding(type_ids)
            + self.side_embedding(side_ids)
            + self.slot_embedding(slot_ids)
        )
