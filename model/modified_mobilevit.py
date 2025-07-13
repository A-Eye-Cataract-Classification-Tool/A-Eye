import torch
import torch.nn as nn
from .transformer_block import TransformerBlock

class ModifiedMobileViT(nn.Module):
    def __init__(self, in_dim=9, embed_dim=192, num_classes=1):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim)
        self.transformer = TransformerBlock(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )

    def forward(self, tokens, pos_enc):
        x = self.proj(tokens)  # [B, 4, 192]
        x = x + pos_enc        # add positional encoding
        x = self.transformer(x)
        x = x.mean(dim=1)      # global average pooling
        return self.classifier(x)  # output maturity score (0–1)
