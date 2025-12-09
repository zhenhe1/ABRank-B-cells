"""
MLP-based regressor for antibody-antigen binding affinity prediction.

This module uses pre-computed PLM embeddings directly (no graph encoding)
and performs mean pooling followed by MLP regression.
"""
import torch
import torch.nn as nn
from typing import List
from torch import Tensor
from torch_geometric.data import Batch as PygBatch
from torch_geometric.nn import global_mean_pool


class MLPRegressor(nn.Module):
    """
    MLP-based regressor that pools pre-computed embeddings and predicts binding affinity.

    Architecture:
    1. Mean pool antibody embeddings: (N_ab, ab_dim) → (B, ab_dim)
    2. Mean pool antigen embeddings: (N_ag, ag_dim) → (B, ag_dim)
    3. Concatenate: (B, ab_dim + ag_dim)
    4. MLP: (B, ab_dim + ag_dim) → (B, hidden_dims[0]) → ... → (B, 1)

    Args:
        ab_dim: Antibody embedding dimension (AntiBERTy: 512)
        ag_dim: Antigen embedding dimension (ESM-2: 1280)
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability before final layer

    Example:
        >>> regressor = MLPRegressor(ab_dim=512, ag_dim=1280, hidden_dims=[512, 256], dropout=0.2)
        >>> affinity_pred = regressor(x_b, x_g, batch)  # (B, 1)
    """

    def __init__(
        self,
        ab_dim: int = 512,
        ag_dim: int = 1280,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.2
    ):
        super().__init__()
        self.ab_dim = ab_dim
        self.ag_dim = ag_dim
        self.pooling = global_mean_pool

        # Build MLP layers
        input_dim = ab_dim + ag_dim
        layers = []

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
            ])
            prev_dim = hidden_dim

        # Add dropout and final output layer
        layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x_b: Tensor, x_g: Tensor, batch: PygBatch) -> Tensor:
        """
        Forward pass: pool embeddings, concatenate, and predict affinity.

        Args:
            x_b: (N_ab, ab_dim) antibody per-residue embeddings
            x_g: (N_ag, ag_dim) antigen per-residue embeddings
            batch: PygBatch with x_b_batch and x_g_batch indices

        Returns:
            affinity_pred: (B, 1) predicted binding affinity (log Kd)
        """
        # Pool antibody and antigen embeddings separately
        h_b = self.pooling(x_b, batch.x_b_batch)  # (B, ab_dim)
        h_g = self.pooling(x_g, batch.x_g_batch)  # (B, ag_dim)

        # Concatenate pooled representations
        h = torch.cat([h_b, h_g], dim=1)  # (B, ab_dim + ag_dim)

        # Pass through MLP
        affinity_pred = self.mlp(h)  # (B, 1)

        return affinity_pred
