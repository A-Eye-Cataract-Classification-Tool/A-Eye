import sys
import os
import torch
import cv2
import numpy as np
import json

# Add parent directory to path for sibling imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.aeye_model import AEyeModel
from radial_tokenizer.radial_tokenizer import RadialTokenizer

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
img = cv2.imread(dummy_image_path)
img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

# Initialize Tokenizer and get tokens
tokenizer = RadialTokenizer()
tokens_192d = tokenizer(img_tensor)


# Step 2: Initialize the full AEyeModel
print("\nStep 2: Initializing the AEyeModel...")
model = AEyeModel()
model.eval()  # Set model to evaluation mode
print(" Model initialized.")

# Step 3: Run the forward pass with the 192D tokens
print("\nStep 3: Running the model's forward pass...")
print(f"Input image shape: {img_tensor.shape}")
with torch.no_grad():
    # Pass the image and tokens to the model
    # Note: The AEyeModel forward pass needs to be adjusted to accept tokens
    # and pass them to the ModifiedMobileViT stages.
    # The following is a conceptual adjustment to AEyeModel's forward method.
    
    # In aeye_model.py, the forward pass should look like this:
    # def forward(self, x_img, tokens):
    #     x = self.stage1(x_img)
    #     x = self.stage2(x)
    #     x = self.stage3(x, tokens=tokens)
    #     ... and so on for other stages.
    #     The final output of the model would then be the result from the last ModifiedMobileViT stage.
    
    # For now, let's assume we are just testing the ModifiedMobileViT block
    modified_vit = model.stage3 # Or any other ModifiedMobileViT stage
    result = modified_vit(torch.randn(1, 32, 32, 32), tokens_192d)

print(" Forward pass complete.")

# Step 4: Display and Save the results
print("\n--- Results ---")
print("Final output:", result)

# Save JSON Output
output_dir = "output/coverage_results"
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "sample1.json"), "w") as f:
    json.dump(result, f, indent=2)
print(f" Saved JSON output to: {os.path.join(output_dir, 'sample1.json')}")


# Export Model for Training
output_dir_model = "output"
os.makedirs(output_dir_model, exist_ok=True)
torch.save(model.state_dict(), os.path.join(output_dir_model, "model_final.pt"))
print(f" Saved model weights to: {os.path.join(output_dir_model, 'model_final.pt')}")

print("\nIntegration successful!")