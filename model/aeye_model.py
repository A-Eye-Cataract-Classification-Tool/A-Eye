import torch
import torch.nn as nn
from modified_mobilevit import ModifiedMobileViT
from typing import Dict
import sys
import os

# Allow sibling imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.modified_mobilevit import ModifiedMobileViT
from radial_tokenizer.radial_tokenizer import RadialTokenizer
from postprocessing.estimate_opacity_coverage import estimate_opacity_coverage


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
        
        # The tokenizer is now part of the model itself
        self.tokenizer = RadialTokenizer()

        # --------- CNN-ViT BACKBONE ---------
        self.stage1 = conv_3x3_bn(3, 16, stride=2)
        self.stage2 = MV2Block(16, 32, stride=2)
        self.stage3 = ModifiedMobileViT(in_channels=32, embed_dim=192)
        self.stage4 = MV2Block(32, 64, stride=2)
        self.stage5 = ModifiedMobileViT(in_channels=64, embed_dim=192)
        self.stage6 = MV2Block(64, 96, stride=2)
        self.stage7 = ModifiedMobileViT(in_channels=96, embed_dim=192)

        # --------- CLASSIFICATION HEAD ---------
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(96, 1)

    def forward(self, x_img: torch.Tensor) -> Dict:
        """
        Performs the ENTIRE forward pass from image to final JSON-ready output.
        """
        # 1. Generate radial tokens from the original input image.
        tokens_192d = self.tokenizer(x_img)

        # 2. Pass image and tokens through the hybrid backbone.
        x = self.stage1(x_img)
        x = self.stage2(x)
        x = self.stage3(x, tokens=tokens_192d)
        x = self.stage4(x)
        x = self.stage5(x, tokens=tokens_192d)
        x = self.stage6(x)
        x = self.stage7(x, tokens=tokens_192d)

        # 3. Get the final classification score.
        x = self.pool(x).flatten(1)
        prediction_prob = torch.sigmoid(self.fc(x))

        # 4. Calculate opacity/coverage metrics from the tokens.
        metrics = estimate_opacity_coverage(tokens_192d)[0]
        
        # 5. Combine prediction and metrics into the final JSON output.
        prediction_label = "Mature Cataract" if prediction_prob.item() >= 0.5 else "Immature Cataract"
        
        result = {
            "prediction": prediction_label,
            **metrics 
        }
        
        return result