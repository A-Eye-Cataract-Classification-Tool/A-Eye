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
import matplotlib.pyplot as plt
import torch  # Optional, only needed if saving as .pt

# === Parameters ===
IMAGE_PATH = 'C:/Users/caspe/Downloads/test/test_image4.jpg'  # Replace with actual image path
CENTER = (64, 64)
RINGS = [(0, 5), (5, 20), (20, 30), (30, 40)]  # Rings
SAVE_PT = True  # Set False if you want .npy instead

# === Load and preprocess image ===
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Image not found at {IMAGE_PATH}")
image = cv2.resize(image, (128, 128))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# === Helper to create ring mask ===
def create_ring_mask(shape, center, inner_r, outer_r):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, outer_r, 255, thickness=-1)
    cv2.circle(mask, center, inner_r, 0, thickness=-1)
    return mask

# === Process each ring ===
all_features = []
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # For visualization
overlay = image_rgb.copy()

for i, (r_in, r_out) in enumerate(RINGS):
    mask = create_ring_mask(image_rgb.shape, CENTER, r_in, r_out)
    masked = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    pixels = masked[mask == 255]  # shape: (num_pixels, 3)

    if pixels.size == 0:
        features = np.zeros(9)
    else:
        mean_rgb = pixels.mean(axis=0)
        std_rgb = pixels.std(axis=0)
        median_rgb = np.median(pixels, axis=0)
        features = np.concatenate([mean_rgb, std_rgb, median_rgb])  # shape: (9,)
    
    all_features.append(features)

    # Draw ring for visualization
    cv2.circle(overlay, CENTER, r_out, colors[i], thickness=2)

# === Stack features ===
tokens = np.stack(all_features, axis=0)  # [4, 9]
tokens = np.expand_dims(tokens, axis=0)  # [1, 4, 9]

# === Save output ===
if SAVE_PT:
    torch.save(torch.tensor(tokens), 'radial_tokens.pt')
else:
    np.save('radial_tokens.npy', tokens)

# === Save visualization ===
plt.figure(figsize=(4, 4))
plt.imshow(overlay)
plt.title("Ring Patches Overlay")
plt.axis('off')
plt.tight_layout()
plt.savefig("patches_visual.png")
print("✅ radial_tokens and patches_visual saved.")
