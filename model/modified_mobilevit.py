"""
====================================================================
NOTE:
This file is part of the initial A-Eye prototype. Implementations
are subject to refinement and may be updated for optimization,
modularity, or alignment with project objectives.
====================================================================
"""

import torch
import torch.nn as nn
from .transformer_block import TransformerBlock

class ModifiedMobileViT(nn.Module):
    """
    Simplified MobileViT variant for radial token input.

    Inputs:
        - tokens: [B, 4, 9] tensor of radial ring feature vectors
        - pos_enc: [B, 4, 192] sinusoidal positional encodings

    Architecture:
        1. Project 9D tokens to 192D
        2. Add radial positional encoding
        3. Pass through Transformer block
        4. Global average pooling
        5. MLP classifier head with sigmoid activation
    """

    def __init__(self, in_dim=9, embed_dim=192, num_classes=1):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim)            # [B, 4, 9] → [B, 4, 192]
        self.transformer = TransformerBlock(embed_dim)      # Self-attention block
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.sigmoid = nn.Sigmoid()                         # separate for flexibility (e.g., during training)

    def forward(self, tokens, pos_enc, return_logits=False):
        """
        Args:
            tokens: [B, 4, 9]
            pos_enc: [B, 4, 192]
            return_logits: If True, returns raw output before sigmoid

        Returns:
            [B, 1] sigmoid probability by default, or logits if specified
        """
        assert tokens.shape[0] == pos_enc.shape[0], "Batch sizes must match"
        assert tokens.shape[1] == pos_enc.shape[1], "Token counts must match"
        x = self.proj(tokens)           # [B, 4, 192]
        x = x + pos_enc                 # add positional encoding
        x = self.transformer(x)         # attention block
        x = x.mean(dim=1)               # global average over radial tokens
        logits = self.classifier(x)     # output scalar per sample
        return logits if return_logits else self.sigmoid(logits)
