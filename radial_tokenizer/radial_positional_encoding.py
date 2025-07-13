"""
====================================================================
NOTE:
This file is part of the initial A-Eye prototype. Implementations
are subject to refinement and may be updated for optimization,
modularity, or alignment with project objectives.
====================================================================
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

# --- Sinusoidal encoding for a single normalized radius ---
def get_sinusoidal_encoding(radius: float, dim: int = 192) -> np.ndarray:
    """
    Computes sinusoidal positional encoding for a single radius value.

    Args:
        radius (float): Normalized radial position in range [0, 1].
        dim (int): Dimensionality of the encoding vector (must be even).

    Returns:
        np.ndarray: A 1D array of shape [dim] containing the positional encoding.
    """
    pe = np.zeros(dim)
    for i in range(dim // 2):
        angle = radius / (10000 ** (2 * i / dim))
        pe[2 * i] = np.sin(angle)
        pe[2 * i + 1] = np.cos(angle)
    return pe

# --- Generate encodings for all rings ---
def get_radial_positional_encoding(num_rings: int = 4, dim: int = 192) -> np.ndarray:
    """
    Generates sinusoidal positional encodings for all radial rings.

    This implementation assumes fixed normalized positions for a simplified
    4-ring radial tokenization setup aligned with a pupil-based image.

    Args:
        num_rings (int): Number of concentric radial rings.
        dim (int): Embedding dimension per ring.

    Returns:
        np.ndarray: A tensor of shape [1, num_rings, dim]
                    suitable for adding to radial token embeddings.
    """
    radii = np.linspace(0, 1, num=num_rings)                        # Normalized radial distances
    encodings = [get_sinusoidal_encoding(r, dim) for r in radii]
    return np.expand_dims(np.stack(encodings), axis=0)              # Shape: [1, num_rings, dim]

# --- Plot and save visualization of the encodings ---
def plot_encodings(encodings: np.ndarray, output_path: str = "encoding_visual.png"):
    """
    Plots the positional encodings for visual inspection.

    Args:
        encodings (np.ndarray): Tensor of shape [1, num_rings, dim]
        output_path (str): Path to save the PNG visualization.
    """
    plt.figure(figsize=(10, 4))
    for i in range(encodings.shape[1]):
        plt.plot(encodings[0][i], label=f"Ring {i+1}")
    plt.title("Radial Positional Encodings (Sinusoidal)")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Encoding Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# --- Main block: runs if script is executed directly ---
if __name__ == "__main__":
    # Settings
    num_rings = 4
    dim = 192

    # Generate the encoding tensor
    enc = get_radial_positional_encoding(num_rings=num_rings, dim=dim)
    print(f"Generated radial positional encoding with shape: {enc.shape}")
    print(f"Normalized ring positions used: {np.linspace(0, 1, num=num_rings)}")

    # Save as .pt (PyTorch tensor)
    torch_enc = torch.tensor(enc, dtype=torch.float32)
    torch.save(torch_enc, "radial_pos_enc.pt")
    print("Saved encoding tensor to: radial_pos_enc.pt")

    # Also save as .npy (NumPy format) for inspection
    np.save("radial_pos_enc.npy", enc)
    print("Saved encoding tensor to: radial_pos_enc.npy")

    # Create and save a line plot
    plot_encodings(enc)
    print("Saved encoding plot to: encoding_visual.png")