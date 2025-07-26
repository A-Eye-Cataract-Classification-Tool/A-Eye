import torch
import torch.nn as nn

# Helper for Modified mobile vit
class TransformerBlock(nn.Module):
    """
    Basic Transformer Encoder block:
    - Multihead Self-Attention
    - FeedForward with residuals
    """
    #ffim and droupout are hard coded
    def __init__(self, dim, heads=4, ff_dim=384, dropout=0.1): #embed dim is dim, num head is head
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        # Multihead self attention
        # x shape: [B, P, d] where B is batch size, P
        attn_output, _ = self.attn(x, x, x)
        
        # Applies residual connection and normalization
        # x shape: [B, P, d]
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward network
        # x shape: [B, P, d]    
        ff_output = self.ff(x)
        
        # Applies residual connection and normalization
        return self.norm2(x + self.dropout(ff_output))
