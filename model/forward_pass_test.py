import sys
import os
import torch
import cv2
import numpy as np

# Add parent directory to path for sibling imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.aeye_model import AEyeModel
from radial_tokenizer.radial_tokenizer import radial_tokenizer

# --- Helper to create a dummy image for testing ---
def create_dummy_image(path="dummy_test_image.png"):
    if not os.path.exists(path):
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.circle(img, (64, 64), 30, (150, 150, 150), -1)
        cv2.imwrite(path, img)
    return path

# --- Main Test Workflow ---

# Step 1: Run the radial tokenizer
print("Step 1: Running the radial tokenizer...")
dummy_image_path = create_dummy_image()

# Correctly unpack the tuple returned by the function
# We get both the projected 192D and original 9D tokens
_, tokens_9d = radial_tokenizer(
    image_path=dummy_image_path,
    output_name="sample1" # Added to match original function signature
)

# Step 2: Prepare a dummy image tensor for the CNN part of the model
x_img = torch.randn(1, 3, 128, 128)  # [Batch, Channels, Height, Width]

# Step 3: Initialize the full AEyeModel
print("\nStep 2: Initializing the AEyeModel...")
model = AEyeModel()
model.eval()  # Set model to evaluation mode
print("✅ Model initialized.")

# Step 4: Run the forward pass with the 9D tokens
print("\nStep 3: Running the model's forward pass...")
print(f"Input image shape: {x_img.shape}")
with torch.no_grad():
    # Pass the 9D tokens to the model
    output = model(x_img, tokens=tokens_9d)
print("✅ Forward pass complete.")

# Step 5: Display the results
print("\n--- Results ---")
print("Final output shape:", output.shape)
print("Predicted value:", output.squeeze())
print("\nIntegration successful!")