import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from aeye_model import AEyeModel

# Step 1: Dummy input tokens and position encodings
x_img = torch.randn(1, 3, 128, 128)      # [B, C=3, H, W]   128x128
tokens = torch.randn(1, 4, 9)        # [Batch, 4 radial rings, 9 features]
pos_enc = torch.randn(1, 4, 192)     # [Batch, 4 rings, 192 positional features]

# Step 2: Initialize the model
model = AEyeModel() 

# Step 3: Run forward pass
output = model(x_img, tokens, pos_enc)

# Step 4: Show output
print("Output shape:", output.shape)
print("Predicted value:", output)