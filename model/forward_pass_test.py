# In A-Eye/model/forward_pass_test.py

import torch
import json
import os
import cv2
import numpy as np
import sys

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.aeye_model import AEyeModel

# --- Main Test Workflow ---
print("🚀 Starting FULL model pipeline test...")

# 1. Initialize the complete, self-contained AEyeModel
model = AEyeModel()
model.eval()
print("✅ AEyeModel initialized.")


# 2. Load your image and prepare the tensor using an absolute path
image_path = r"C:\Users\itsrr\Documents\PUP\AEYE\aeye\A-Eye\model\dummy_test_img.jpg" # Use the full, exact path
img = cv2.imread(image_path)

# Check if the image was loaded correctly
if img is None:
    raise FileNotFoundError(f"Image not found at '{image_path}'. Please double-check the full path.")

# Resize the image to 128x128 and prepare the tensor
img = cv2.resize(img, (128, 128))
img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
print(f"✅ Loaded and prepared '{image_path}' with shape: {img_tensor.shape}")


# 3. Run the full forward pass with a single call
with torch.no_grad():
    final_output = model(img_tensor) # This one line runs the whole pipeline!

print("\n--- ✅ Forward Pass Complete ---")
print("Final JSON Output:")
print(json.dumps(final_output, indent=4))

# 4. Save the deliverables
output_dir = "output"
os.makedirs(os.path.join(output_dir, "coverage_results"), exist_ok=True)

# Save JSON output
json_path = os.path.join(output_dir, "coverage_results", "example_output.json")
with open(json_path, "w") as f:
    json.dump(final_output, f, indent=4)
print(f"\n💾 Saved example JSON to: {json_path}")

# Save the trained model weights
model_path = os.path.join(output_dir, "model_final.pt")
torch.save(model.state_dict(), model_path)
print(f"💾 Saved model weights to: {model_path}")

print("\n🎉 Integration successful! Model is ready for training.")