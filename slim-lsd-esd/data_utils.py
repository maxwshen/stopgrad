import numpy as np
import torch
import torchvision.datasets as datasets
from training import distributed_mode
from training.data_transform import get_transform_cifar, get_transform_mnist
import models.rng as rng 
from functools import partial

def get_data_loader(args, is_for_fid, logger):

    if is_for_fid:
        bsz = args.batch_size_sampling_fid
    else:
        bsz = args.batch_size

    # Serialize first-time downloads across distributed workers.
    if not distributed_mode.is_main_process():
        distributed_mode.barrier()

    if args.dataset == "cifar10":
        transforms = get_transform_cifar(is_for_fid)
        dataset = datasets.CIFAR10(
            root=args.data_path,
            train=True, # in gen models, fid always computed on training set.
            download=True,
            transform=transforms,
        )
    elif args.dataset == "mnist":  # 3x32x32 MNIST for fast development
        transforms = get_transform_mnist()
        dataset = datasets.MNIST(
            root=args.data_path,
            train=True,
            download=True,
            transform=transforms,
        )
    else:
        raise NotImplementedError(f"Unsupported dataset {args.dataset}")

    if distributed_mode.is_main_process():
        distributed_mode.barrier()

    logger.info(dataset)

    logger.info("Initializing DataLoader")
    num_tasks = distributed_mode.get_world_size()
    global_rank = distributed_mode.get_rank()
    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        worker_init_fn=partial(rng.worker_init_fn, rank=global_rank),
        batch_size=bsz,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        persistent_workers=True,
        drop_last=not is_for_fid,  # for FID evaluation, we want to keep all samples
    )
    logger.info(str(sampler))
    return data_loader
