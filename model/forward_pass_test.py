import torch
from modified_mobilevit import ModifiedMobileViT
import matplotlib.pyplot as plt
import numpy as np

# Helper to run and debug model
def test_forward_pass(tokens, pos_enc):
    try:
        model = ModifiedMobileViT()
        output = model(tokens, pos_enc)
        print("Forward pass successful. Output:", output)
    except AssertionError as e:
        print("Assertion error:", e)
    except Exception as ex:
        print("Unexpected error:", ex)

# Valid input
print(" Test 1: Valid Input")
tokens = torch.randn(1, 4, 9)
pos_enc = torch.randn(1, 4, 192)
test_forward_pass(tokens, pos_enc)

# Invalid input (wrong shape)
print("\n Test 2: Invalid Positional Encoding Shape")
bad_pos_enc = torch.randn(1, 5, 192)  # Should trigger assertion
test_forward_pass(tokens, bad_pos_enc)

# Visualize the 192D token embeddings (example from projected tokens)
sample_embedding = pos_enc[0, 0].detach().numpy()
plt.figure(figsize=(8, 3))
plt.plot(sample_embedding)
plt.title("Ring 1 Positional Encoding")
plt.xlabel("Embedding Dimension")
plt.ylabel("Value")
plt.grid(True)
plt.tight_layout()
plt.show()