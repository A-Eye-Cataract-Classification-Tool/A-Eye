"""
====================================================================
NOTE:
This file is part of the initial A-Eye prototype. Implementations
are subject to refinement and may be updated for optimization,
modularity, or alignment with project objectives.
====================================================================
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import os

# ========= Helper Functions =========

def get_pupil_center(mask):
    """
    Compute the centroid of the pupil from a binary mask.
    If mask is invalid or empty, return image center as fallback.
    """
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return (64, 64)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def create_ring_mask(shape, center, inner_r, outer_r):
    """
    Create a binary ring mask centered at 'center' with given inner and outer radii.
    """
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, outer_r, 255, -1)
    cv2.circle(mask, center, inner_r, 0, -1)
    return mask


def extract_ring_features(image, mask):
    """
    Extract 9D features (mean, std, median) from masked region.
    """
    masked = cv2.bitwise_and(image, image, mask=mask)
    pixels = masked[mask == 255]
    if pixels.shape[0] == 0:
        return np.zeros(9)
    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)
    median = np.median(pixels, axis=0)
    return np.concatenate([mean, std, median])  # shape: (9,)


def draw_ring_overlay(image, center, rings, colors):
    """
    Overlay colored rings on image.
    """
    overlay = image.copy()
    for i, (_, r_out) in enumerate(rings):
        cv2.circle(overlay, center, r_out, colors[i], thickness=2)
    return overlay


# ========= Main Tokenization Function =========

def radial_tokenize(image_path, mask_path=None, output_prefix="radial_tokens"):
    """
    Main function to compute radial tokens from an eye image.
    Saves both 9D and projected 192D tensors + ring overlay.
    """
    print(f"🖼️  Processing: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = cv2.resize(image, (128, 128))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load mask if provided
    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (128, 128))
        center = get_pupil_center(mask)
    else:
        center = (64, 64)

    # Define 4 concentric rings
    rings = [(0, 20), (20, 40), (40, 60), (60, 80)]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    # Extract 9D features per ring
    features = [extract_ring_features(image_rgb, create_ring_mask(image.shape, center, r0, r1))
                for (r0, r1) in rings]
    tokens_9d = np.stack(features, axis=0)     # [4, 9]
    tokens_9d = np.expand_dims(tokens_9d, 0)   # [1, 4, 9]

    # Project 9D to 192D
    projector = nn.Linear(9, 192)
    tokens_9d_tensor = torch.tensor(tokens_9d, dtype=torch.float32)
    tokens_192d = projector(tokens_9d_tensor)  # [1, 4, 192]

    # Save tensors
    torch.save(tokens_9d_tensor, f"{output_prefix}_9D.pt")
    torch.save(tokens_192d, f"{output_prefix}_192D.pt")

    # Save visualization
    overlay = draw_ring_overlay(image_rgb, center, rings, colors)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{output_prefix}_visual.png", overlay_bgr)

    print(f"✅ Saved: {output_prefix}_9D.pt, {output_prefix}_192D.pt, {output_prefix}_visual.png")


# ========= Run Example =========

if __name__ == "__main__":
    # Replace with your actual image & mask paths
    radial_tokenize(
        image_path="C:/Users/caspe/Downloads/test/test_image.jpg",
        mask_path=None,  # or provide "C:/path/to/binary_mask.png"
        output_prefix="output/radial_tokens"
    )