"""
====================================================================
NOTE:
This file is part of the A-Eye prototype. This version is optimized
for training by saving only essential outputs, organized by type.
Now includes a visual overlay of rings for preview.
====================================================================
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import os


# ===================== Helper Functions ===================== #

def get_pupil_center(mask):
    """
    Compute centroid of pupil from binary mask.
    If mask empty or invalid, fallback to (64, 64) for 128x128 image.
    """
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return (64, 64)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def create_ring_mask(shape, center, inner_r, outer_r):
    """
    Create binary mask for a ring defined by inner and outer radius.
    """
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, outer_r, 255, -1)
    cv2.circle(mask, center, inner_r, 0, -1)
    return mask


def extract_ring_features(image, mask):
    """
    Extract 9D feature vector: mean, std, median RGB from masked region.
    """
    masked = cv2.bitwise_and(image, image, mask=mask)
    pixels = masked[mask == 255]
    if pixels.shape[0] == 0:
        return np.zeros(9)
    mean_rgb = pixels.mean(axis=0)
    std_rgb = pixels.std(axis=0)
    median_rgb = np.median(pixels, axis=0)
    return np.concatenate([mean_rgb, std_rgb, median_rgb])  # shape: (9,)


def draw_ring_overlay(image_rgb, center, rings):
    """
    Return a copy of the image with colored rings drawn for visualization.
    """
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    overlay = image_rgb.copy()
    for i, (_, r_out) in enumerate(rings):
        cv2.circle(overlay, center, r_out, colors[i], thickness=2)
    return overlay


# ===================== Main Function ===================== #

def radial_tokenize(image_path, mask_path=None, output_base="output", output_name="sample", save_9d=False):
    """
    Extract radial tokens from a pupil-centered 128x128 eye image.
    Saves [1,4,192] projected tokens, and optionally [1,4,9] raw tokens.
    Also saves a preview image with 4 rings drawn.
    """
    # --- Load image ---
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.resize(image, (128, 128))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- Get center from mask or fallback ---
    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (128, 128))
        center = get_pupil_center(mask)
    else:
        center = (64, 64)

    # --- Define uniform rings for 128x128 image ---
    rings = [(0, 16), (16, 32), (32, 48), (48, 64)]

    # --- Extract raw 9D tokens ---
    features = [
        extract_ring_features(image_rgb, create_ring_mask(image.shape, center, r0, r1))
        for (r0, r1) in rings
    ]
    tokens_9d = np.expand_dims(np.stack(features, axis=0), 0)  # shape: [1, 4, 9]
    tokens_9d_tensor = torch.tensor(tokens_9d, dtype=torch.float32)

    # --- Project 9D → 192D ---
    projector = nn.Linear(9, 192)
    tokens_192d = projector(tokens_9d_tensor)  # shape: [1, 4, 192]

    # --- Save outputs ---
    os.makedirs(f"{output_base}/tokens_192D", exist_ok=True)
    torch.save(tokens_192d, f"{output_base}/tokens_192D/{output_name}.pt")

    if save_9d:
        os.makedirs(f"{output_base}/tokens_9D", exist_ok=True)
        torch.save(tokens_9d_tensor, f"{output_base}/tokens_9D/{output_name}.pt")

    # --- Save preview overlay ---
    overlay_rgb = draw_ring_overlay(image_rgb, center, rings)
    os.makedirs(f"{output_base}/visuals", exist_ok=True)
    cv2.imwrite(f"{output_base}/visuals/{output_name}.png",
                cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

    print(f"✅ Saved: {output_base}/tokens_192D/{output_name}.pt" +
          (f" and tokens_9D/{output_name}.pt" if save_9d else "") +
          f" | Preview: {output_base}/visuals/{output_name}.png")


# ===================== Script Entry Point ===================== #
if __name__ == "__main__":
    # Example usage — replace with your test image paths
    radial_tokenize(
        image_path="C:/Users/caspe/Downloads/test/test1.png",  # <-- change to your actual image
        mask_path=None,
        output_base="output",
        output_name="test1",
        save_9d=True
    )
