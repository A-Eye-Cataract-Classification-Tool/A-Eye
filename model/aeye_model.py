import torch
import torch.nn as nn
from modified_mobilevit import ModifiedMobileViT

# CONV 3X3
def conv_3x3_bn(inp, oup, stride=1):
    """3x3 convolution + BatchNorm + SiLU activation"""
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

# MobileNetV2 
class MV2Block(nn.Module):
    """MobileNetV2 inverted residual block"""
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        hidden_dim = int(inp * expansion)
        self.use_res_connect = stride == 1 and inp == oup
        self.conv = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)


# A-Eye model architecture
class AEyeModel(nn.Module):
    def __init__(self):
        super().__init__()

        # --------- BONE ---------
        self.stage1 = conv_3x3_bn(3, 16, stride=2)                             # Stage 1: 128x128x3 → 64x64x16
        self.stage2 = MV2Block(16, 32, stride=2)                               # Stage 2: 64x64x16 → 32x32x32
        self.stage3 = ModifiedMobileViT(in_channels=32, embed_dim=192)        # Stage 3: Radial + Global

        self.stage4 = MV2Block(32, 64, stride=2)                               # Stage 4: 32x32x32 → 16x16x64
        self.stage5 = ModifiedMobileViT(in_channels=64, embed_dim=192)        # Stage 5: Deep Radial

        # --------- NECK ---------
        self.stage6 = MV2Block(64, 96, stride=2)                               # Stage 6: 16x16x64 → 8x8x96
        self.stage7 = ModifiedMobileViT(in_channels=96, embed_dim=192)        # Stage 7: High-level Radial

        # --------- HEAD ---------
        self.pool = nn.AdaptiveAvgPool2d((1, 1))                               # Stage 8: Global pooling
        self.fc = nn.Linear(96, 1)                                             # Stage 9: Binary maturity score

    def forward(self, x_img, tokens=None, pos_enc=None):
        print("\nInput image shape:", x_img.shape)
        if tokens is not None:
            print("\nRadial tokens shape:", tokens.shape)
        if pos_enc is not None:
            print("\nRadial positional encoding shape:", pos_enc.shape)

        print("\n=================STAGES==================")
        x = self.stage1(x_img)
        print("\nAfter stage1:", x.shape)
        x = self.stage2(x)
        print("\nAfter stage2:", x.shape)
        
        x = self.stage3(x, tokens=tokens, pos_enc=pos_enc)  # Only pass once
        print("\nAfter stage3:", x.shape)

        x = self.stage4(x)
        print("\nAfter stage4:", x.shape)
        
        # These don't use tokens/pos_enc for now
        x = self.stage5(x)
        print("\nAfter stage5:", x.shape)

        x = self.stage6(x)
        print("\nAfter stage6:", x.shape)

        x = self.stage7(x)
        print("\nAfter stage7:", x.shape)

        x = self.pool(x).flatten(1)
        print("\nAfter global pooling:", x.shape)

        out = torch.sigmoid(self.fc(x))
        print("\nFinal output shape:", out.shape)

        return out
