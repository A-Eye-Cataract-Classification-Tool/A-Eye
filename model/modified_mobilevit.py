import torch
import torch.nn as nn
from transformer_block import TransformerBlock
from radial_tokenizer_stub import RadialTokenizer  # <-- stub
from radial_positional_encoding_stub import RadialPositionEmbedding  # <-- stub


class ModifiedMobileViT(nn.Module):
    """
    MobileViT block with radial tokenizer and transformer integration.
    Stages follow the theoretical architecture for local + global fusion.
    """
    def __init__(self, in_channels=32, embed_dim=192, num_heads=2):
        super().__init__()

        self.local_conv = nn.Sequential(                            # Step 1
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )

        self.proj_in = nn.Conv2d(in_channels, embed_dim, kernel_size=1)  # Step 2
        self.token_proj = nn.Linear(9, embed_dim)  # Project 9D tokens to embedding dim

        self.tokenizer = RadialTokenizer()                          # Step 3
        self.pos_encoder = RadialPositionEmbedding()               # Step 4

        self.transformer = TransformerBlock(embed_dim, num_heads)  # Step 5

        self.proj_out = nn.Conv2d(embed_dim, in_channels, kernel_size=1)  # Step 7

        self.fuse = nn.Sequential(                                  # Step 9
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )

    def forward(self, x, tokens=None, pos_enc=None):
        res = x  # Save input for skip connection

        x = self.local_conv(x)              # Step 1
        x = self.proj_in(x)                 # Step 2 → [B, D, H, W]

        if tokens is None:
            tokens, meta = self.tokenizer(x)  # Step 3 → [B, P, D]
        else:
            tokens = self.token_proj(tokens)  # [B, 4, 9] → [B, 4, 192]
            meta = {
                'ring_count': tokens.shape[1],
                'original_shape': x.shape[2:]
            }

        if pos_enc is None:
            pe = self.pos_encoder(meta, x.device, batch_size=x.shape[0])  # Step 4
        else:
            pe = pos_enc

        tokens = tokens + pe                # Step 4
        tokens = self.transformer(tokens)   # Step 5

        x = self.tokenizer.fold(tokens, meta, x.shape[2:])  # Step 6 → [B, D, H, W]
        x = self.proj_out(x)                # Step 7 → [B, C, H, W]

        x = x + res                         # Step 8: Residual
        x = self.fuse(x)                    # Step 9
        return x                            # Step 10
