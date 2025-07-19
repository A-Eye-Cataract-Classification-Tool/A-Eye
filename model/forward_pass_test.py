import torch
from modified_mobilevit import ModifiedMobileViT

# Step 1: Dummy input tokens and position encodings
tokens = torch.randn(1, 4, 9)        # [Batch, 4 radial rings, 9 features]
pos_enc = torch.randn(1, 4, 192)     # [Batch, 4 rings, 192 positional features]

# Step 2: Initialize the model
model = ModifiedMobileViT()

# Step 3: Run forward pass
output = model(tokens, pos_enc)

# Step 4: Show output
print("Output shape:", output.shape)
print("Predicted value:", output)