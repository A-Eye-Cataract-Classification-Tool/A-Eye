"""
====================================================================
NOTE:
This file is part of the initial A-Eye prototype. Implementations
are subject to refinement and may be updated for optimization,
modularity, or alignment with project objectives.
====================================================================
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
CENTER = (64, 64)                                   # Assumed pupil center in 128x128 image
RINGS = [(0, 20), (20, 40), (40, 60), (60, 80)]     # 4 fixed-radius concentric zones
IMG_SIZE = (128, 128)
DEVICE = torch.device('cpu')

# --- Create circular mask for a ring using Euclidean distance (Equation 1) ---
def create_ring_mask(shape, center, inner_r, outer_r):
    """
    Create a binary mask where pixels within the given inner and outer radius are set to 1.
    This approximates radial distance computation: r = sqrt((x - x0)^2 + (y - y0)^2)
    """
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, outer_r, 255, thickness=-1)
    cv2.circle(mask, center, inner_r, 0, thickness=-1)
    return mask

# --- Extract features per ring: mean, std, median RGB values (total 9D per ring) ---
def extract_radial_tokens(image, center=CENTER, rings=RINGS):
    """
    Divides the image into fixed concentric circular patches centered on the pupil.
    Each patch is summarized by 9D feature vector (mean, std, median of RGB).
    """
    features = []
    for inner, outer in rings:
        mask = create_ring_mask(image.shape, center, inner, outer)
        masked = cv2.bitwise_and(image, image, mask=mask)
        pixels = masked[mask == 255]
        if len(pixels) == 0:
            features.append(np.zeros(9))
        else:
            mean = pixels.mean(axis=0)
            std = pixels.std(axis=0)
            median = np.median(pixels, axis=0)
            features.append(np.concatenate([mean, std, median]))
    tokens = np.stack(features)          # Shape: [4, 9]
    return tokens[np.newaxis, ...]       # Shape: [1, 4, 9]

# --- Visualize ring boundaries for debugging/documentation ---
def visualize_rings(image, center=CENTER, rings=RINGS, output_path='patches_visual.png'):
    vis = image.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for i, (inner, outer) in enumerate(rings):
        cv2.circle(vis, center, outer, colors[i], thickness=2)
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title("Radial Token Zones (4 Patches)")
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()

# --- Main test block ---
if __name__ == "__main__":
    image_path = "test_image.png"
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.resize(image, IMG_SIZE)

    # Extract 4 ring tokens
    tokens = extract_radial_tokens(image)  # [1, 4, 9]
    print("Extracted token shape:", tokens.shape)

    # Save as .pt file
    torch_tokens = torch.tensor(tokens, dtype=torch.float32).to(DEVICE)
    torch.save(torch_tokens, "radial_tokens.pt")
    print("Saved token tensor to radial_tokens.pt")

    # Save visual
    visualize_rings(image)
    print("Saved ring overlay to patches_visual.png")