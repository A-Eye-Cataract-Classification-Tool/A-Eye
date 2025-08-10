"""
====================================================================
NOTE:
This file is part of the A-Eye prototype. This version is optimized
for training by saving only essential outputs, organized by type.
====================================================================
"""

import cv2
import torch
import numpy as np
from torch import nn

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

# ========= Radial Tokenizer Module (Corrected) =========
class RadialTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        # The internal projector has been REMOVED.
        self.center = (64, 64)
        self.rings = [(0, 16), (16, 32), (32, 48), (48, 64)]

    def forward(self, image_tensor):
        B = image_tensor.shape[0]
        device = image_tensor.device

        tokens_9d_list = []
        for img in image_tensor:
            img_np = img.permute(1, 2, 0).cpu().numpy() * 255.0
            img_np = img_np.astype(np.uint8)

            ring_features = []
            for r0, r1 in self.rings:
                mask = create_ring_mask(img_np.shape, self.center, r0, r1)
                ring_feat = extract_ring_features(img_np, mask)
                ring_features.append(ring_feat)

            tokens_9d_list.append(ring_features)

        tokens_9d = torch.tensor(tokens_9d_list, dtype=torch.float32, device=device)

        # Return the raw 9D tokens directly. The projection step is gone.
        return tokens_9d