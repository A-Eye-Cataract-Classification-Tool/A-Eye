import torch
import torch.nn as nn
from transformer_block import TransformerBlock
from radial_tokenizer_stub import RadialTokenizer  # Stub: breaks image into radial tokens temporary
from radial_positional_encoding_stub import RadialPositionEmbedding  # Stub: adds radial position info temporary


class ModifiedMobileViT(nn.Module):
    """
    A MobileViT-style block using radial tokenization instead of grid patches.
    Tracks input-output shapes across 10 stages to combine local and global features.
    """

    def __init__(self, in_channels=32, embed_dim=192, num_heads=2):
        super().__init__()

        # Step 1: Local feature extractor (3×3 conv keeps spatial size)
        # [B, C, H, W] -> [B, C, H, W]
        self.local_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )

        # Step 2: Project to transformer input dims (embed dim = d)
        # [B, C, H, W] -> [B, d, H, W]
        self.proj_in = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

        # For manual token input, 9D tokens -> 192D embeddings
        # [B, P, 9] -> [B, P, d]
        self.token_proj = nn.Linear(9, embed_dim)

        # Step 3: Radial tokenization
        # [B, d, H, W] -> [B, P, d], where P = number of radial patches
        self.tokenizer = RadialTokenizer()

        # Step 4: Learnable positional encoding
        # [B, P, d] + [B, P, d] -> [B, P, d]
        self.pos_encoder = RadialPositionEmbedding()

        # Step 5: Transformer to process radial tokens globally
        # [B, P, d] -> [B, P, d]
        self.transformer = TransformerBlock(embed_dim, num_heads)

        # Step 7: Project transformer output back to original CNN channels
        # [B, d, H, W] -> [B, C, H, W]
        self.proj_out = nn.Conv2d(embed_dim, in_channels, kernel_size=1)

        # Step 9: Fuse local + global features via final conv block
        # [B, C, H, W] -> [B, C, H, W]
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )

    def forward(self, x, tokens=None, pos_enc=None):
        """
        Forward pass with optional token & position input override.
        Input:
            x        : [B, C, H, W] - image feature map
            tokens   : [B, P, 9] (optional) - radial token features
            pos_enc  : [B, P, d] (optional) - positional embeddings

        Output:
            [B, C, H, W] - enhanced feature map
        """

        # Save for Step 8 (skip connection)
        res = x  # [B, C, H, W]

        # Step 1: Local 3x3 conv
        x = self.local_conv(x)  # [B, C, H, W]

        # Step 2: Embed to transformer space
        x = self.proj_in(x)     # [B, d, H, W]

        # Step 3: Radial tokenization
        if tokens is None:
            tokens, meta = self.tokenizer(x)  # [B, P, d], meta includes ring count & shape
        else:
            tokens = self.token_proj(tokens)  # [B, P=4, 9] -> [B, P=4, d=192]
            meta = {
                'ring_count': tokens.shape[1],
                'original_shape': x.shape[2:]  # H, W
            }

        # Step 4: Positional encoding
        if pos_enc is None:
            pe = self.pos_encoder(meta, device=x.device, batch_size=x.size(0))  # [B, P, d]
        else:
            pe = pos_enc

        tokens = tokens + pe  # [B, P, d]

        # Step 5: Transformer for global reasoning
        tokens = self.transformer(tokens)  # [B, P, d]

        # Step 6: Fold tokens back to spatial map
        x = self.tokenizer.fold(tokens, meta, x.shape[2:])  # [B, d, H, W]

        # Step 7: Project back to original channel size
        x = self.proj_out(x)  # [B, C, H, W]

        # Step 8: Add skip connection from input
        x = x + res  # [B, C, H, W]

        # Step 9: Final fusion conv
        x = self.fuse(x)  # [B, C, H, W]

        # Step 10: Return fused output
        return x
