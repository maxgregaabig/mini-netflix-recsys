"""Two-tower retrieval model in PyTorch.

Architecture:
- User tower: user_id embedding + aggregated genre affinity → MLP → L2-normalized embedding
- Item tower: item_id embedding + genre multi-hot → MLP → L2-normalized embedding

Training uses BPR (Bayesian Personalized Ranking) loss with in-batch negatives.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    def __init__(self, num_users: int, num_genres: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim, padding_idx=0)
        self.genre_proj = nn.Linear(num_genres, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, user_ids: torch.Tensor, genre_affinity: torch.Tensor) -> torch.Tensor:
        u_emb = self.user_emb(user_ids)
        g_emb = self.genre_proj(genre_affinity)
        combined = torch.cat([u_emb, g_emb], dim=-1)
        out = self.mlp(combined)
        return F.normalize(out, p=2, dim=-1)


class ItemTower(nn.Module):
    def __init__(self, num_items: int, num_genres: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.genre_proj = nn.Linear(num_genres, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, item_ids: torch.Tensor, genre_multihot: torch.Tensor) -> torch.Tensor:
        i_emb = self.item_emb(item_ids)
        g_emb = self.genre_proj(genre_multihot)
        combined = torch.cat([i_emb, g_emb], dim=-1)
        out = self.mlp(combined)
        return F.normalize(out, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, num_genres: int,
                 embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.user_tower = UserTower(num_users, num_genres, embedding_dim, hidden_dim)
        self.item_tower = ItemTower(num_items, num_genres, embedding_dim, hidden_dim)
        self.embedding_dim = embedding_dim

    def forward(
        self,
        user_ids: torch.Tensor,
        user_genre_aff: torch.Tensor,
        item_ids: torch.Tensor,
        item_genre_mh: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        user_emb = self.user_tower(user_ids, user_genre_aff)
        item_emb = self.item_tower(item_ids, item_genre_mh)
        return user_emb, item_emb


def bpr_loss(user_emb: torch.Tensor, pos_item_emb: torch.Tensor,
             neg_item_emb: torch.Tensor) -> torch.Tensor:
    """BPR loss: -log(sigmoid(pos_score - neg_score))."""
    pos_scores = (user_emb * pos_item_emb).sum(dim=-1)
    neg_scores = (user_emb * neg_item_emb).sum(dim=-1)
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def in_batch_bpr_loss(user_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
    """In-batch negatives BPR loss.

    For each user i, item i is positive, all other items j≠i are negatives.
    """
    # Similarity matrix: [B, B]
    scores = torch.mm(user_emb, item_emb.t())
    batch_size = scores.size(0)

    # Positive scores on diagonal
    pos_scores = scores.diag().unsqueeze(1)  # [B, 1]

    # All negative scores
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=scores.device)
    neg_scores = scores[mask].view(batch_size, batch_size - 1)  # [B, B-1]

    # BPR: mean over all pos-neg pairs
    diff = pos_scores - neg_scores  # [B, B-1]
    return -F.logsigmoid(diff).mean()
