"""
====================================================================
NOTE:
This file is part of the A-Eye prototype. This version is optimized
for training by saving only essential outputs, organized by type.
====================================================================
"""

import os
import cv2
import torch
import numpy as np
import logging
import random
from torch import nn

# ========= Reproducibility =========
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ========= Logging =========
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ========= Helper Functions =========

def get_pupil_center(mask):
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return (64, 64)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

def create_ring_mask(shape, center, inner_r, outer_r):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, outer_r, 255, -1)
    cv2.circle(mask, center, inner_r, 0, -1)
    return mask

def extract_ring_features(image, mask):
    masked = cv2.bitwise_and(image, image, mask=mask)
    pixels = masked[mask == 255]
    if pixels.shape[0] == 0:
        return np.zeros(9)
    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)
    median = np.median(pixels, axis=0)
    return np.concatenate([mean, std, median])  # shape: (9,)

# ========= Projector Class =========

class RadialProjector(nn.Module):
    def __init__(self, in_dim=9, out_dim=192):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x)

# ========= Main Function =========

def radial_tokenize(
    image_path,
    mask_path=None,
    output_base="output",
    output_name="sample1",
    projector=None,
    device="cpu",
    save_9d=False
):
    logging.info(f"🖼️  Processing: {image_path}")

    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = cv2.resize(image, (128, 128))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use pupil center from mask or fallback
    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (128, 128))
        center = get_pupil_center(mask)
    else:
        center = (64, 64)

    # Extract radial ring features
    rings = [(0, 20), (20, 40), (40, 60), (60, 80)]
    features = [extract_ring_features(image_rgb, create_ring_mask(image.shape, center, r0, r1)) for (r0, r1) in rings]
    tokens_9d = np.expand_dims(np.stack(features, axis=0), 0)  # [1, 4, 9]

    # Project to 192D
    if projector is None:
        projector = RadialProjector()
    projector = projector.to(device)
    projector.eval()

    tokens_9d_tensor = torch.tensor(tokens_9d, dtype=torch.float32).to(device)
    tokens_192d = projector(tokens_9d_tensor)

    # ========= Save Outputs by Type =========
    os.makedirs(output_base, exist_ok=True)
    path_192d = os.path.join(output_base, "tokens_192D")
    path_proj = os.path.join(output_base, "projection_weights")
    path_9d   = os.path.join(output_base, "tokens_9D") if save_9d else None

    os.makedirs(path_192d, exist_ok=True)
    os.makedirs(path_proj, exist_ok=True)
    if save_9d:
        os.makedirs(path_9d, exist_ok=True)

    torch.save(tokens_192d.detach().cpu(), os.path.join(path_192d, f"{output_name}.pt"))
    torch.save(projector.state_dict(), os.path.join(path_proj, f"{output_name}.pt"))
    if save_9d:
        torch.save(tokens_9d_tensor.cpu(), os.path.join(path_9d, f"{output_name}.pt"))

    # ========= Logs =========
    logging.info("✅ Saved:")
    logging.info(f" - 192D token: {os.path.join(path_192d, output_name + '.pt')}")
    logging.info(f" - Projection weights: {os.path.join(path_proj, output_name + '.pt')}")
    if save_9d:
        logging.info(f" - (Optional) 9D token: {os.path.join(path_9d, output_name + '.pt')}")
        
    # ========= return the tensors =========
    return tokens_192d, tokens_9d_tensor

# ========= Run Example =========

if __name__ == "__main__":
    radial_tokenize(
        image_path="C:/Users/denni/Downloads/test.png",     # update path
        mask_path=None,
        output_base="output",
        output_name="sample1",
        save_9d=False                                       # change to True to create 9D
    )