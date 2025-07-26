"""
====================================================================
NOTE:
This file is part of the initial A-Eye prototype. Implementations
are subject to refinement and may be updated for optimization,
modularity, or alignment with project objectives.

NOTE: 
MobileVit + Radial Tokenization block
====================================================================
"""

# Torch = base library for building deep learning models
import torch
import torch.nn as nn

# transformer block (self-attention + feed-forward)
from transformer_block import TransformerBlock

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

    # sets up the layers and parts of the model
    def __init__(self, in_dim=9, embed_dim=192, num_classes=1):
        super().__init__()

        # Step 1: Project 9D input to 192D embedding (per token)
        self.proj = nn.Linear(in_dim, embed_dim)            # [B, 4, 9] → [B, 4, 192]

        # Step 2: Transformer block ( step 3 multi-head self-attention + step 4 feed forward) calls transformer_block.py
        self.transformer = TransformerBlock(embed_dim)      # Self-attention block

        # Step 5: Classifier head (MLP)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),   # Compress to smaller rep
            nn.ReLU(),                  # Non-linear activation
            nn.Linear(64, num_classes)  # Final output: [B, 1] if binary classification
        )

        # Sigmoid activation to convert logits → probabilities (0 to 1)
        self.sigmoid = nn.Sigmoid()                         # separate for flexibility (e.g., during training)

    # describes how input flows through the model
    def forward(self, tokens, pos_enc, return_logits=False):
        """
        Args:
            tokens: [B, 4, 9]
            pos_enc: [B, 4, 192]
            return_logits: If True, returns raw output before sigmoid

        Returns:
            [B, 1] sigmoid probability by default, or logits if specified
        """

        # Sanity checks to make sure inputs are aligned
        assert tokens.shape[0] == pos_enc.shape[0], "Batch sizes must match"
        assert tokens.shape[1] == pos_enc.shape[1], "Token counts must match"

        # Step 1: Project from 9D to 192D
        x = self.proj(tokens)           # [B, 4, 192]
        print("🔹 After projection:", x.shape)

        #  Step 2: Add positional information
        x = x + pos_enc                 # add positional encoding
        print("🔹 After adding pos_enc:", x.shape)

        #  Step 3: Run through transformer (self-attention)
        x = self.transformer(x)         # attention block
        print("🔹 After transformer:", x.shape)

        #  Step 4: Global average pooling across the 4 tokens
        x = x.mean(dim=1)               # global average over radial tokens
        print("🔹 After global average pooling:", x.shape)

        #  Step 5: Classify using MLP
        logits = self.classifier(x)     # output scalar per sample
        print("🔹 Before sigmoid (logits):", logits.shape)

        # return logits or sigmoid output
        return logits if return_logits else self.sigmoid(logits)
