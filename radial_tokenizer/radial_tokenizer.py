import cv2
import numpy as np

def create_ring_mask(shape, center, inner_r, outer_r):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, outer_r, 255, -1)
    cv2.circle(mask, center, inner_r, 0, -1)
    return mask

def extract_radial_tokens(image, center=(64, 64), rings=[(0,20), (20,40), (40,60), (60,80)]):
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
    tokens = np.stack(features)         # shape [4, 9]
    return tokens[np.newaxis, ...]      # shape [1, 4, 9]
