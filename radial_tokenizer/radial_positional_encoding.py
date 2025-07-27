"""
====================================================================
NOTE:
This file is part of the A-Eye prototype. This refined version applies
learnable positional embeddings to radial tokens and saves only essential
outputs under /output for model integration.
====================================================================
"""

import os
import torch
import torch.nn as nn
import logging

# === Setup logging ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class RadialPositionEmbedding(nn.Module):
    """
    Learnable positional embeddings for concentric radial tokens.

    Args:
        num_rings (int): Number of concentric rings (default: 4).
        embed_dim (int): Dimensionality of each token vector (default: 192).
    """

    def __init__(self, num_rings: int = 4, embed_dim: int = 192):
        super().__init__()
        self.num_rings = num_rings
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_embeddings=num_rings, embedding_dim=embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional embeddings to radial tokens.

        Args:
            x (Tensor): Input tensor of shape (B, 4, 192)

        Returns:
            Tensor: Positionally encoded tokens (B, 4, 192)
        """
        B, num_tokens, dim = x.shape
        assert num_tokens == self.num_rings, f"Expected {self.num_rings} tokens, got {num_tokens}"
        assert dim == self.embed_dim, f"Expected embedding dim {self.embed_dim}, got {dim}"

        indices = torch.arange(self.num_rings, device=x.device).unsqueeze(0).expand(B, -1)
        pos_embed = self.embedding(indices)
        return x + pos_embed


# ========================== Standalone Execution ==========================
if __name__ == "__main__":
    input_path = "output/tokens_192D/sample1.pt"
    output_dir = "output/encoded_tokens"
    output_file = os.path.join(output_dir, "encoded_radial_tokens.pt")

    # Logging start
    logging.info(f"📂  Processing: {input_path}")

    # Validate input
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input token file not found: {input_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Load and encode
    tokens = torch.load(input_path)  # [B, 4, 192]
    encoder = RadialPositionEmbedding()
    encoded = encoder(tokens)

    # Save output
    torch.save(encoded, output_file)

    # Logging end
    logging.info("✅ Saved:")
    logging.info(f" - Encoded tokens: {output_file}")