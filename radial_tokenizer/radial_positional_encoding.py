"""
====================================================================
NOTE:
This file is part of the initial A-Eye prototype. Implementations
are subject to refinement and may be updated for optimization,
modularity, or alignment with project objectives.
====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class RadialPositionEmbedding(nn.Module):
    """
    Applies learnable positional embeddings to radial tokens.
    Each concentric ring receives a distinct embedding vector.
    """

    def __init__(self, num_rings: int = 4, embed_dim: int = 192):
        """
        Args:
            num_rings (int): Number of concentric radial rings.
            embed_dim (int): Dimensionality of each token vector.
        """
        super().__init__()
        self.num_rings = num_rings
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_embeddings=num_rings, embedding_dim=embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Radial tokens of shape (batch_size, num_rings, embed_dim)

        Returns:
            Tensor: Positionally encoded tokens of same shape.
        """
        batch_size, num_tokens, dim = x.shape
        assert num_tokens == self.num_rings, f"Expected {self.num_rings} tokens, got {num_tokens}."
        assert dim == self.embed_dim, f"Expected embed_dim {self.embed_dim}, got {dim}."

        # Create [0, 1, 2, ..., num_rings - 1] and expand for batch
        ring_indices = torch.arange(self.num_rings, device=x.device).unsqueeze(0)
        ring_indices = ring_indices.expand(batch_size, -1)

        pos_embed = self.embedding(ring_indices)  # (batch_size, num_rings, embed_dim)
        return x + pos_embed


# ========================== DEBUG / STANDALONE ==========================
if __name__ == "__main__":
    print("Testing RadialPositionEmbedding module...\n")

    # FOR DEBUGGING
    batch_size = 5              # Batch size for testing
    num_rings = 4               # Number of concentric rings
    embed_dim = 192             # Dimensionality of each token vector

    # Simulate dummy radial token embeddings
    dummy_input = torch.randn(batch_size, num_rings, embed_dim)
    print(f"Input shape: {dummy_input.shape}")

    # Initialize and apply radial position encoding
    pos_encoder = RadialPositionEmbedding(num_rings=num_rings, embed_dim=embed_dim)
    output = pos_encoder(dummy_input)

    print(f"Output shape: {output.shape}")
    print("✓ Learnable positional encoding applied successfully.")
    print("\nPreview of encoded tensor:")

    # Shows the full content of the positional embedding tensor
    print(output)

    # Prints the shape of each ring embedding vector for every batch
    for batch_idx in range(output.shape[0]):
        print(f"\nBatch {batch_idx + 1}:")
        for ring_idx in range(output.shape[1]):
            vector = output[batch_idx, ring_idx]
            print(f"  Ring {ring_idx + 1} vector shape: {vector.shape}")

    # Save the output tensor as .pt and .npy
    torch.save(output, "radial_pos_enc.pt")
    np_output = output.detach().cpu().numpy()
    np.save("radial_pos_enc.npy", np_output)
    print("\nSaved: radial_pos_enc.pt and radial_pos_enc.npy")

    # Visualize the embeddings as line plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    for ring in range(num_rings):
        plt.plot(np_output[0, ring], label=f"Ring {ring + 1}")
    plt.title("Learnable Positional Embeddings per Ring")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("encoding_visual.png")
    print("Saved: encoding_visual.png")
