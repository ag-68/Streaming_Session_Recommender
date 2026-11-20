"""SASRec: read the liked films in order and score the next one.

The backbone follows Kang and McAuley (2018). Training uses full-catalog
cross-entropy, so the training question matches the full-catalog test.

When side features are supplied, each film vector is
E'_j = E_j + W g_j, used for both the sequence input and the output scores.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        batch, length, _ = hidden.shape
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)
        query = query.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        query = query.masked_fill(pad[:, None, :, None], 0.0)
        key = key.masked_fill(pad[:, None, :, None], 0.0)
        value = value.masked_fill(pad[:, None, :, None], 0.0)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.triu(torch.ones(length, length, device=hidden.device, dtype=torch.bool), 1)
        scores = scores.masked_fill(causal, float("-inf"))
        fully_blocked = torch.isneginf(scores).all(dim=-1, keepdim=True)
        scores = scores.masked_fill(fully_blocked, 0.0)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        context = (weights @ value).transpose(1, 2).contiguous().view(batch, length, -1)
        return self.out(context).masked_fill(pad.unsqueeze(-1), 0.0)


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class Block(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = CausalSelfAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = FeedForward(hidden_dim, dropout)
        self.norm_attention = nn.LayerNorm(hidden_dim)
        self.norm_feed_forward = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        hidden = hidden.masked_fill(pad.unsqueeze(-1), 0.0)
        attended = self.attention(hidden, pad)
        hidden = self.norm_attention(hidden + self.dropout(attended))
        hidden = hidden.masked_fill(pad.unsqueeze(-1), 0.0)
        transformed = self.feed_forward(hidden).masked_fill(pad.unsqueeze(-1), 0.0)
        hidden = self.norm_feed_forward(hidden + transformed)
        return hidden.masked_fill(pad.unsqueeze(-1), 0.0)


class SASRec(nn.Module):
    def __init__(
        self,
        n_items: int,
        max_len: int = 100,
        hidden_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout: float = 0.2,
        side_dim: int = 0,
        use_polarity: bool = False,
    ) -> None:
        super().__init__()
        self.n_items = n_items
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.side_dim = side_dim
        self.use_polarity = use_polarity
        self.item_emb = nn.Embedding(n_items + 1, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        nn.init.normal_(self.item_emb.weight, std=1 / math.sqrt(hidden_dim))
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            Block(hidden_dim, num_heads, dropout) for _ in range(num_blocks)
        )
        if use_polarity:
            # 0 = dislike, 1 = like
            self.polarity_emb = nn.Embedding(2, hidden_dim)
            nn.init.normal_(self.polarity_emb.weight, std=0.02)
        else:
            self.polarity_emb = None
        if side_dim > 0:
            self.side_proj = nn.Linear(side_dim, hidden_dim, bias=False)
            nn.init.zeros_(self.side_proj.weight)
            self.register_buffer(
                "side_features",
                torch.zeros(n_items + 1, side_dim),
                persistent=True,
            )
        else:
            self.side_proj = None
            self.register_buffer("side_features", torch.zeros(n_items + 1, 1), persistent=False)

    def set_side_features(self, features: torch.Tensor) -> None:
        if self.side_proj is None:
            raise ValueError("this model was built without side features")
        if features.shape != (self.n_items + 1, self.side_dim):
            raise ValueError(
                f"expected side features {(self.n_items + 1, self.side_dim)}, got {tuple(features.shape)}"
            )
        self.side_features = features.detach().to(dtype=self.side_features.dtype)

    def item_vectors(self) -> torch.Tensor:
        """Tied film vectors used for sequence input and catalog scoring."""

        if self.side_proj is None:
            return self.item_emb.weight
        return self.item_emb.weight + self.side_proj(self.side_features)

    def forward(
        self,
        sequences: torch.Tensor,
        polarities: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a score for every catalog id at every position.

        sequences uses 0 as the empty slot and is left-padded to max_len.
        polarities uses 0 = dislike and 1 = like (ignored on pad slots).
        """

        return self.encode(sequences, polarities) @ self.item_vectors().T

    def encode(
        self,
        sequences: torch.Tensor,
        polarities: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The film representation at each position, before scoring the catalog."""

        pad = sequences == 0
        positions = torch.arange(sequences.size(1), device=sequences.device)
        vectors = self.item_vectors()
        hidden = vectors[sequences] * math.sqrt(self.hidden_dim)
        if self.polarity_emb is not None:
            if polarities is None:
                polarities = torch.ones_like(sequences)
            hidden = hidden + self.polarity_emb(polarities.clamp(0, 1))
        hidden = hidden + self.pos_emb(positions).unsqueeze(0)
        hidden = self.emb_dropout(hidden).masked_fill(pad.unsqueeze(-1), 0.0)
        for block in self.blocks:
            hidden = block(hidden, pad)
        return hidden
