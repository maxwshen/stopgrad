"""
This module provides utilities to mimic JAX's random number generator (RNG) control in PyTorch,
enabling reproducible and distributed-safe random operations.
However, GPU computation is not strictly deterministic, so full numerical reproducibility is not guaranteed.
"""

import torch
import hashlib
import random
import numpy as np


def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def fold_in(seed: int, *args) -> int:
    """
    Simulate jax.random.fold_in via SHA256 hashing.
    Args can be anything hashable: step, rank, etc.
    """
    data = str((seed,) + args)
    h = hashlib.sha256(data.encode("utf-8")).hexdigest()
    folded_seed = int(h, 16) % (2**63)  # Safe for torch.manual_seed()
    return folded_seed



def worker_init_fn(worker_id, rank, base_seed=0):
    seed = fold_in(base_seed, worker_id, rank, "worker_init")
    torch.manual_seed(fold_in(seed, 'torch'))
    random.seed(fold_in(seed, 'python'))
    np.random.seed(fold_in(seed, 'numpy') % 2**32)
