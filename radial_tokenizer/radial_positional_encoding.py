import numpy as np

def get_sinusoidal_encoding(radius, dim=192):
    pe = np.zeros(dim)
    for i in range(dim // 2):
        angle = radius / (10000 ** (2 * i / dim))
        pe[2 * i] = np.sin(angle)
        pe[2 * i + 1] = np.cos(angle)
    return pe

def get_radial_positional_encoding(num_rings=4, dim=192):
    radii = np.linspace(0, 1, num=num_rings)
    encodings = [get_sinusoidal_encoding(r, dim) for r in radii]
    return np.expand_dims(np.stack(encodings), axis=0)  # shape [1, 4, 192]
