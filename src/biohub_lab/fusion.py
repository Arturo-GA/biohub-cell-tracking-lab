"""Division-aware reverse association; a hypothesis, not a trained model.

The public harmonic baseline normalizes each target over possible parents.
Its reverse association is a learned model evaluated on reversed inputs,
which can disagree around mitosis. We reduce that reverse contribution only
for geometrically plausible, well-supported two-daughter candidates.
"""
import torch


def division_aware_reverse_weight(
    forward_prob, source_zyx, target_zyx, base_weight=0.15,
    min_parent_prob=0.55, full_parent_prob=0.85,
    max_parent_um=9.0, max_sister_um=14.0,
    voxel_scale=(1.625, 0.40625, 0.40625), strength=1.0,
):
    """Return [batch, source, 1] reverse weights, never an extra edge.

    Coordinates are in ORIGINAL voxels. Decisions use predictions only.
    A sigmoid/softmax-normalized probability tensor of shape [B,S,T] is
    expected, with sum over S equal to one. Empty sets are supported.
    """
    if not 0 <= base_weight <= 1 or not 0 <= strength <= 1:
        raise ValueError("weights must be in [0, 1]")
    if not 0 <= min_parent_prob < full_parent_prob <= 1:
        raise ValueError("invalid probability ramp")
    if forward_prob.ndim != 3:
        raise ValueError("forward_prob must be [batch, source, target]")
    b, s, t = forward_prob.shape
    if source_zyx.shape != (b, s, 3) or target_zyx.shape != (b, t, 3):
        raise ValueError("coordinate shapes do not match probabilities")
    if not torch.isfinite(forward_prob).all():
        raise ValueError("non-finite probabilities")
    result = forward_prob.new_full((b, s, 1), base_weight)
    if s == 0 or t < 2 or strength == 0:
        return result
    scale = forward_prob.new_tensor(voxel_scale)
    top_p, top_j = forward_prob.topk(2, dim=-1)
    batch = torch.arange(b, device=forward_prob.device)[:, None, None]
    daughters = target_zyx[batch, top_j]
    distance = ((daughters - source_zyx[:, :, None]) * scale).norm(dim=-1)
    sister_distance = ((daughters[:, :, 0] - daughters[:, :, 1]) * scale).norm(dim=-1)
    geometry_ok = (distance <= max_parent_um).all(dim=-1) & (sister_distance <= max_sister_um)
    ramp = ((top_p[..., 1] - min_parent_prob) / (full_parent_prob - min_parent_prob)).clamp(0, 1)
    protection = strength * ramp * geometry_ok.to(ramp.dtype)
    return result * (1 - protection[..., None])
