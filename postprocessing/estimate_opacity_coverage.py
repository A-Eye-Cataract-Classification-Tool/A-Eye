import torch

def estimate_opacity_coverage(tokens_192d: torch.Tensor, threshold: float = 0.5):
    """
    Estimate opacity and coverage based on token brightness.

    Args:
        tokens_192d (Tensor): shape [B, 4, 192]
        threshold (float): threshold for coverage decision

    Returns:
        List[Dict]: One dict per batch item, each with:
        {
            "opacity_percent": float,
            "coverage_percent": float
        }
    """
    results = []

    opacity = tokens_192d.mean(dim=(1, 2)) * 100  # [B]
    ring_mean = tokens_192d.mean(dim=2)           # [B, 4]
    coverage = (ring_mean > threshold).sum(dim=1) / 4.0 * 100  # [B]

    for o, c in zip(opacity, coverage):
        results.append({
            "opacity_percent": round(o.item(), 2),
            "coverage_percent": round(c.item(), 2)
        })

    return results