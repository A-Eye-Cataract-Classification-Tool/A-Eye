import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.modified_mobilevit import ModifiedMobileViT
from radial_tokenizer.radial_tokenizer import RadialTokenizer

def conv_3x3_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        hidden_dim = int(inp * expansion)
        self.use_res_connect = stride == 1 and inp == oup
        self.conv = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden_dim), nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.SiLU(),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup),
        )
    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)

class AEyeModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # This layer processes the 3-channel input image before tokenization
        self.local_representation = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=1)
        )
        # The tokenizer is now owned by the main model
        self.tokenizer = RadialTokenizer()

        # Model Stages
        self.stage1 = conv_3x3_bn(3, 16, stride=2)
        self.stage2 = MV2Block(16, 32, stride=2)
        self.stage3 = ModifiedMobileViT(in_channels=32, embed_dim=192)
        self.stage4 = MV2Block(32, 64, stride=2)
        self.stage5 = ModifiedMobileViT(in_channels=64, embed_dim=192)
        self.stage6 = MV2Block(64, 96, stride=2)
        self.stage7 = ModifiedMobileViT(in_channels=96, embed_dim=192)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(96, 1)

    def forward(self, x_img):
        print(f"Initial input shape: {x_img.shape}")

        # 1. Apply local representation to the 3-channel input image
        x_local_rep = self.local_representation(x_img)
        
        # 2. Generate tokens ONCE from the locally processed 3-channel image
        tokens = self.tokenizer(x_local_rep)
        print(f"Generated Tokens Shape: {tokens.shape}")

        # 3. Proceed with the main model architecture
        x = self.stage1(x_img)
        print(f"After Stage 1: {x.shape}")
        x = self.stage2(x)
        print(f"After Stage 2: {x.shape}")
        
        # Pass the pre-computed tokens to the MobileViT stages
        x = self.stage3(x, tokens=tokens)
        print(f"After Stage 3: {x.shape}")
        x = self.stage4(x)
        print(f"After Stage 4: {x.shape}")
        x = self.stage5(x, tokens=tokens)
        print(f"After Stage 5: {x.shape}")
        x = self.stage6(x)
        print(f"After Stage 6: {x.shape}")
        x = self.stage7(x, tokens=tokens)
        print(f"After Stage 7: {x.shape}")

        x = self.pool(x).flatten(1)
        print(f"After Pooling: {x.shape}")
        out = torch.sigmoid(self.fc(x))
        print(f"Final Output: {out.shape}")
        return out
