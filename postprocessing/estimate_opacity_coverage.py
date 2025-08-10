# postprocessing/estimate_opacity_coverage.py
import torch
from typing import List, Dict

def estimate_opacity_coverage(tokens_192d: torch.Tensor, threshold: float = 0.5) -> List[Dict]:
    """
    Estimates opacity and coverage based on token brightness from radial patches.

    Args:
        tokens_192d (torch.Tensor): The token tensor from the tokenizer, with a 
                                    shape of [B, 4, 192]. 'B' is the batch size, 
                                    '4' represents the four radial rings, and 
                                    '192' is the feature dimension for each token.
        threshold (float): A value between 0 and 1. If a ring's average brightness 
                           is above this, it's considered "covered" by the cataract.

    Returns:
        List[Dict]: A list of dictionaries, one for each item in the batch. Each dict contains:
                    {
                        "opacity_percent": float,
                        "coverage_percent": float
                    }
    """
    results = []
    
    # Ensure the input tensor is a float for calculations
    tokens_192d = tokens_192d.float()

    # Opacity: The overall average brightness of all tokens across all rings.
    # We multiply by 100 to express it as a percentage.
    # Shape change: [B, 4, 192] -> mean across dims 1 and 2 -> [B]
    opacity = tokens_192d.mean(dim=(1, 2)) * 100

    # Coverage: How many of the 4 rings are "significantly" opaque.
    # 1. Get the average brightness for each of the 4 rings.
    # Shape change: [B, 4, 192] -> mean across dim 2 -> [B, 4]
    ring_mean = tokens_192d.mean(dim=2)
    
    # 2. Check which rings exceed the brightness threshold.
    # Shape change: [B, 4] -> boolean tensor of same shape
    covered_rings = ring_mean > threshold
    
    # 3. Calculate the percentage of covered rings.
    # Shape change: [B, 4] -> sum across dim 1 -> [B] -> divide by 4.0 and multiply by 100
    coverage = covered_rings.sum(dim=1) / 4.0 * 100

    # Format the results for each item in the batch
    for o, c in zip(opacity, coverage):
        results.append({
            "opacity_percent": round(o.item(), 2),
            "coverage_percent": round(c.item(), 2)
        })

    return results