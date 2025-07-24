import torch
import torch.nn as nn

class RadialPositionEmbedding(nn.Module):
    
    def forward(self, meta, device, batch_size=1):
        P = meta['ring_count']
        D = 192  # still hardcoded
        return torch.randn(batch_size, P, D, device=device)