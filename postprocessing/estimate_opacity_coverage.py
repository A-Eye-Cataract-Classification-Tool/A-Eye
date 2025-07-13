import numpy as np

def estimate_opacity_percent(tokens):  # tokens: [4, 9]
    mean_rgb = tokens[:, :3].mean(axis=1)  # mean per ring
    brightness = mean_rgb.mean()
    return round((brightness / 255) * 100, 2)

def estimate_coverage_percent(tokens, threshold=180):
    mean_rgb = tokens[:, :3].mean(axis=1)
    is_opaque = (mean_rgb > threshold).astype(int)
    return int((is_opaque.sum() / 4) * 100)
