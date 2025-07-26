"""
====================================================================
NOTE:
This file is part of the initial A-Eye prototype. Implementations
are subject to refinement and may be updated for optimization,
modularity, or alignment with project objectives.
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
    return np.concatenate([mean, std, median])

def draw_ring_overlay(image, center, rings, colors):
    overlay = image.copy()
    for i, (_, r_out) in enumerate(rings):
        cv2.circle(overlay, center, r_out, colors[i], thickness=2)
    return overlay

# ========= Projector Class =========

class RadialProjector(nn.Module):
    def __init__(self, in_dim=9, out_dim=192):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x)

# ========= Main Function =========

def radial_tokenize(image_path, mask_path=None, output_prefix="output/radial_tokens", projector=None, device="cpu"):
    logging.info(f"🖼️  Processing: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = cv2.resize(image, (128, 128))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (128, 128))
        center = get_pupil_center(mask)
    else:
        center = (64, 64)

    rings = [(0, 20), (20, 40), (40, 60), (60, 80)]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    features = [extract_ring_features(image_rgb, create_ring_mask(image.shape, center, r0, r1)) for (r0, r1) in rings]
    tokens_9d = np.stack(features, axis=0)        # [4, 9]
    tokens_9d = np.expand_dims(tokens_9d, 0)      # [1, 4, 9]

    # Prepare projector
    if projector is None:
        projector = RadialProjector()
    projector = projector.to(device)
    projector.eval()

    tokens_9d_tensor = torch.tensor(tokens_9d, dtype=torch.float32).to(device)
    tokens_192d = projector(tokens_9d_tensor)

    # Save outputs
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    torch.save(tokens_9d_tensor.cpu(), f"{output_prefix}_9D.pt")
    torch.save(tokens_192d.detach().cpu(), f"{output_prefix}_192D.pt")
    torch.save(projector.state_dict(), f"{output_prefix}_projection_weights.pt")

    torch.save({
        "features_9D": tokens_9d_tensor.cpu(),
        "features_192D": tokens_192d.detach().cpu(),
        "center": center,
        "image_path": image_path
    }, f"{output_prefix}_metadata.pt")

    overlay = draw_ring_overlay(image_rgb, center, rings, colors)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{output_prefix}_visual.png", overlay_bgr)

    logging.info(f"✅ Saved: {output_prefix}_9D.pt, {output_prefix}_192D.pt, {output_prefix}_visual.png")
    logging.info(f"📦 Metadata and projection weights saved for training reproducibility.")

# ========= Run Example =========

if __name__ == "__main__":
    radial_tokenize(
        image_path="C:/Users/denni/Downloads/test.png",            # Change this path
        mask_path=None,
        output_prefix="output/radial_tokens"
    )