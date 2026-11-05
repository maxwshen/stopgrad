import math
import wandb
import argparse
import torch
import torch.nn as nn
import numpy as np

import argparse
import gc
import logging
import math
from typing import Iterable, Any, Callable
import time

from torchvision.utils import make_grid

from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.aggregation import MeanMetric
import torch.distributed as dist


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x, nrow=nrow, normalize=True, value_range=(-1,1))
    x = x.mul(255).add_(0.5).clamp_(0,255).permute(1,2,0).to('cpu', torch.uint8).numpy()
    return x

def wrap_wandb(arr):                                                                                                                                                 
    return wandb.Image(array2grid(arr))

def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }



def print_model(model, logger):
    logger.info("=" * 91)
    num_params = 0
    for name, param in model.named_parameters():
        param_std = param.std().item()
        if param.requires_grad:
            num_params += param.numel()
            logger.info(f"{name:48} | {str(list(param.shape)):24} | std: {param_std:.6f}")
    logger.info("=" * 91)
    logger.info(f"Total params: {num_params}")


'''
def synchronize_gradients(model: torch.nn.Module):
    #In a distributed setting, to enable jvp, we need to call model.module instead of model directly.
    #If so, we synchronize gradients across all processes.
    if not isinstance(model, DistributedDataParallel):
        return
    torch.cuda.synchronize()
    for param in model.module.parameters():
        if param.requires_grad and param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= dist.get_world_size()
'''

def _gradient_buckets(parameters, bucket_cap_mb):
    """Yield dense gradient buckets without reordering parameters."""
    if bucket_cap_mb <= 0:
        raise ValueError("bucket_cap_mb must be positive")

    bucket_cap_bytes = int(bucket_cap_mb * 1024 * 1024)
    bucket = []
    bucket_bytes = 0
    bucket_device = None
    bucket_dtype = None

    for p in parameters:
        grad = p.grad
        if not p.requires_grad or grad is None:
            continue
        if grad.is_sparse:
            raise NotImplementedError("Sparse gradients are not supported by manual synchronization")

        grad_bytes = grad.numel() * grad.element_size()
        incompatible = (
            bucket
            and (grad.device != bucket_device or grad.dtype != bucket_dtype)
        )
        would_overflow = bucket and (bucket_bytes + grad_bytes > bucket_cap_bytes)
        if incompatible or would_overflow:
            yield bucket
            bucket = []
            bucket_bytes = 0

        if not bucket:
            bucket_device = grad.device
            bucket_dtype = grad.dtype
        bucket.append(grad)
        bucket_bytes += grad_bytes

    if bucket:
        yield bucket


def synchronize_gradients(model, bucket_cap_mb=25):
    if not isinstance(model, DistributedDataParallel):
        return

    world_size = dist.get_world_size()
    for grads in _gradient_buckets(model.module.parameters(), bucket_cap_mb):
        flat_grads = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
        flat_grads.div_(world_size)
        synced_grads = torch._utils._unflatten_dense_tensors(flat_grads, grads)
        for grad, synced_grad in zip(grads, synced_grads):
            grad.copy_(synced_grad)


def gradient_sanity_check(model):
    if not isinstance(model, DistributedDataParallel):
        return
    torch.cuda.synchronize()
    # logging.info(f"Gradient sanity check ...")
    for name, p in model.module.named_parameters():
        if p.requires_grad and len(p.shape) > 3:
            monitor = p.grad.norm()

            monitor_list = [torch.zeros_like(monitor) for _ in range(dist.get_world_size())]
            dist.all_gather(monitor_list, monitor)
            monitor_tensor = torch.stack(monitor_list)
            # logging.info(f"All_gathered grad norm, param {name}: ")
            # for i, m in enumerate(monitor_tensor):
            #     logging.info(f"Rank {i}: {m:.16f}")
            # break

            # Assert all gradient norms are close to rank 0's
            ref = monitor_tensor[0]
            for i, m in enumerate(monitor_tensor):
                assert torch.isclose(m, ref), \
                    f"Gradient norm mismatch at rank {i}: {m} vs rank 0: {ref}"


def get_compiled_counts():
    metrics = torch._dynamo.utils.get_compilation_metrics()
    return len(metrics)
