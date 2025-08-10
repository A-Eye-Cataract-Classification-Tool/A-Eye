import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path to allow sibling imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.transformer_block import TransformerBlock
from radial_tokenizer.radial_positional_encoding import RadialPositionEmbedding

# Placeholder for folding tokens back to a grid.
def fold_tokens_to_grid(tokens, output_size):
    """Placeholder function to map radial tokens back to a spatial grid."""
    B, P, D = tokens.shape
    H, W = output_size
    # Dummy implementation: broadcast the first token across the grid
    return tokens[:, 0].unsqueeze(-1).unsqueeze(-1).expand(B, D, H, W)


class ModifiedMobileViT(nn.Module):
    """
    A MobileViT-style block that accepts 9D radial tokens and projects them internally.
    """
    def __init__(self, in_channels=32, embed_dim=192, num_heads=2):
        super().__init__()

        # Local feature extractor
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )
        self.proj_in = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

        # Learnable positional encoding for the radial tokens
        self.pos_encoder = RadialPositionEmbedding(num_rings=4, embed_dim=embed_dim)

        # Transformer to process radial tokens
        self.transformer = TransformerBlock(embed_dim, num_heads)

        # Project transformer output back to CNN channel dimensions
        self.proj_out = nn.Conv2d(embed_dim, in_channels, kernel_size=1)

        # Fuse local and global features
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )

    def forward(self, x, tokens):
        """
        Forward pass with 9D radial tokens.
        Args:
            x (torch.Tensor): [B, C, H, W] - image feature map.
            tokens (torch.Tensor): [B, P, 9] - 9-feature radial tokens.
        """
        res = x
        x_local = self.local_conv(x)
        x_proj = self.proj_in(x_local)

        # Apply positional encoding directly to the 192D tokens
        tokens_encoded = self.pos_encoder(tokens)

        # Apply transformer for global reasoning
        tokens_transformed = self.transformer(tokens_encoded)

        # Fold tokens back to a spatial map
        x_global = fold_tokens_to_grid(tokens_transformed, output_size=x_proj.shape[2:])

        # Project back to original channel size and add skip connection
        x_global = self.proj_out(x_global)
        x = x_global + res

        # Final fusion
        x = self.fuse(x)
        return x