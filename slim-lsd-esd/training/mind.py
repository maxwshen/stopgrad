"""Monge Inception Distance helpers."""

import torch
import torch.distributed as dist

from training import distributed_mode


def gather_features(features: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Gather equal-sized per-rank feature shards and trim distributed padding."""
    if distributed_mode.get_world_size() == 1:
        return features[:num_samples]

    gathered = [torch.empty_like(features) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, features)
    return torch.cat(gathered, dim=0)[:num_samples]


def compute_mind(
    generated_features: torch.Tensor,
    reference_features: torch.Tensor,
    rng_seed: int = 0,
    n_projections: int = 100,
) -> torch.Tensor:
    """Compute MIND from equal-sized Inception feature arrays."""
    if generated_features.ndim != 2 or reference_features.ndim != 2:
        raise ValueError("MIND expects two-dimensional feature arrays")
    if generated_features.shape != reference_features.shape:
        raise ValueError(
            "MIND expects matching feature shapes, got "
            f"{tuple(generated_features.shape)} and {tuple(reference_features.shape)}"
        )
    if n_projections <= 0:
        raise ValueError("n_projections must be positive")

    generated_features = generated_features.float()
    reference_features = reference_features.float()
    num_samples, feature_dim = generated_features.shape

    generator = torch.Generator(device=generated_features.device)
    generator.manual_seed(rng_seed)
    projections = torch.randn(
        n_projections,
        feature_dim,
        generator=generator,
        device=generated_features.device,
        dtype=generated_features.dtype,
    )
    projections = torch.nn.functional.normalize(projections, dim=1)

    generated_projected = projections @ generated_features.T
    reference_projected = projections @ reference_features.T
    generated_projected = generated_projected.sort(dim=1).values
    reference_projected = reference_projected.sort(dim=1).values
    projection_distances = (
        (generated_projected - reference_projected).square().mean(dim=1)
    )
    return 3 * feature_dim * projection_distances.mean()
