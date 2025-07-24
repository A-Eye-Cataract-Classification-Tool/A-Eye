import torch
import torch.nn as nn

class RadialTokenizer(nn.Module):
    def forward(self, x):
        B, D, H, W = x.shape
        P = 4  # number of radial tokens
        dummy_tokens = torch.randn(B, P, D, device=x.device)
        meta = {'ring_count': P, 'original_shape': (H, W)}
        return dummy_tokens, meta

    def fold(self, tokens, meta, output_size):
        B, P, D = tokens.shape
        H, W = output_size
        # Just expand tokens[0] across the grid as a dummy filler
        expanded = tokens[:, 0].unsqueeze(-1).unsqueeze(-1).expand(B, D, H, W)
        return expanded
