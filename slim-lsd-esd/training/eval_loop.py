# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import gc
import logging
import math
import os
from argparse import Namespace
from pathlib import Path
from typing import Iterable

import PIL.Image

import torch
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image
from training import distributed_mode
import models.rng as rng

logger = logging.getLogger(__name__)

PRINT_FREQUENCY = 5


class PreemptionRequested(Exception):
    """Raised at an evaluation batch boundary to save and exit promptly."""


def _extract_inception_features(
    fid_metric: FrechetInceptionDistance,
    images: torch.Tensor,
) -> torch.Tensor:
    if fid_metric.normalize and not fid_metric.used_custom_model:
        images = (images * 255).byte()
    features = fid_metric.inception(images)
    if features.ndim == 1:
        features = features.unsqueeze(0)
    return features


def create_fid_metric(device: torch.device) -> FrechetInceptionDistance:
    """Create one metric per rank while serializing the initial weights download."""
    if distributed_mode.is_main_process():
        fid_metric = FrechetInceptionDistance(
            normalize=True,
            reset_real_features=False,
        ).to(device=device, non_blocking=True)
    if distributed_mode.is_dist_avail_and_initialized():
        distributed_mode.barrier()
    if not distributed_mode.is_main_process():
        fid_metric = FrechetInceptionDistance(
            normalize=True,
            reset_real_features=False,
        ).to(device=device, non_blocking=True)
    return fid_metric


@torch.no_grad()
def eval_model(
    model: DistributedDataParallel,
    net_eval: torch.nn.Module,
    nfe: int,
    data_loader: Iterable,
    device: torch.device,
    update_steps: int,
    epoch_num: int,
    args: Namespace,
    fid_samples: int,
    overfit_samples = None,
    overfit_labels=None,
    fid_metric: FrechetInceptionDistance | None = None,
    compute_real_features: bool = True,
    feature_samples: int = 0,
    collect_real_features: bool = False,
    collect_fake_features: bool = False,
    compute_fid_score: bool = True,
    should_stop=None,
):
    gc.collect()
    model.train(False)
    net_eval.eval()


    assert args.compute_fid or args.compute_mind

    if args.distributed:
        data_loader.sampler.set_epoch(0)

    assert fid_samples <= len(data_loader.dataset), (
        f"In this interface, dataset size ({len(data_loader.dataset)}) must be larger than FID samples ({fid_samples})."
    )
    local_fid_samples = math.ceil(fid_samples / distributed_mode.get_world_size())

    if fid_metric is None:
        fid_metric = create_fid_metric(device)
    # reset_real_features=False preserves cached real statistics and clears fake ones.
    fid_metric.reset()

    num_synthetic = 0
    num_real = 0
    real_feature_chunks = []
    fake_feature_chunks = []
    #snapshots_saved = False
    #if args.fid_dir:
    #    (Path(args.fid_dir) / "snapshots").mkdir(parents=True, exist_ok=True)

    for data_iter_step, (samples, _) in enumerate(data_loader):

        if should_stop is not None and should_stop():
            raise PreemptionRequested

        if num_synthetic >= local_fid_samples:
            break

        shape_to_sample = samples.shape 

        if args.overfit and (overfit_samples is not None):
            samples = overfit_samples
            print('setting fid batch to overfit batch')
            shape_to_sample = (
                args.batch_size_sampling_fid, samples.shape[1], samples.shape[2], samples.shape[3]
            )

        if compute_real_features:
            real_samples = samples[: local_fid_samples - num_real]
            real_samples = real_samples.to(device, non_blocking=True)
            if compute_fid_score:
                fid_metric.update(real_samples, real=True)
            if collect_real_features:
                features_needed = feature_samples - sum(
                    chunk.shape[0] for chunk in real_feature_chunks
                )
                if features_needed > 0:
                    real_features = _extract_inception_features(
                        fid_metric,
                        real_samples[:features_needed],
                    )
                    real_feature_chunks.append(real_features.detach())
            num_real += real_samples.shape[0]

        if num_synthetic < local_fid_samples:          
            model_without_ddp = model.module if isinstance(model, DistributedDataParallel) else model  

            with torch.random.fork_rng(devices=[device]):
                #per node and per step seed
                torch.manual_seed(rng.fold_in(args.seed, rng.get_rank(), data_iter_step, epoch_num, update_steps))
                with torch.amp.autocast('cuda', enabled=False), torch.no_grad():
                    synthetic_samples = model_without_ddp.sample(samples_shape=shape_to_sample, net=net_eval, device=device, nfe=nfe)
            torch.cuda.synchronize()

            # Scaling to [0, 1] from [-1, 1]
            synthetic_samples = torch.clamp(
                synthetic_samples * 0.5 + 0.5, min=0.0, max=1.0
            )
            synthetic_samples = torch.floor(synthetic_samples * 255)

            synthetic_samples = synthetic_samples.to(torch.float32) / 255.0

            if num_synthetic + synthetic_samples.shape[0] > local_fid_samples:
                synthetic_samples = synthetic_samples[: local_fid_samples - num_synthetic]
            if compute_fid_score:
                fid_metric.update(synthetic_samples, real=False)
            if collect_fake_features:
                features_needed = feature_samples - sum(
                    chunk.shape[0] for chunk in fake_feature_chunks
                )
                if features_needed > 0:
                    fake_features = _extract_inception_features(
                        fid_metric,
                        synthetic_samples[:features_needed],
                    )
                    fake_feature_chunks.append(fake_features.detach())
            num_synthetic += synthetic_samples.shape[0]
            
            #if not snapshots_saved and args.fid_dir:
            #    save_image(
            #        synthetic_samples,
            #        fp=Path(args.fid_dir)
            #        / "snapshots"
            #        / f"{epoch}_{data_iter_step}{suffix}.png",
            #    )
            #    snapshots_saved = True

            #if args.save_fid_samples and args.fid_dir:
            #    images_np = (
            #        (synthetic_samples * 255.0)
            #        .clip(0, 255)
            #        .to(torch.uint8)
            #        .permute(0, 2, 3, 1)
            #        .cpu()
            #        .numpy()
            #    )
            #    for batch_index, image_np in enumerate(images_np):
            #        image_dir = Path(args.fid_dir) / f"fid_samples{suffix}"
            #        os.makedirs(image_dir, exist_ok=True)
            #        image_path = (
            #            image_dir
            #            / f"{distributed_mode.get_rank()}_{data_iter_step}_{batch_index}.png"
            #        )
            #        PIL.Image.fromarray(image_np, "RGB").save(image_path)

        #if not args.compute_fid:
        #    return {}

        if (
            compute_fid_score
            and (
                (data_iter_step + 1) % PRINT_FREQUENCY == 0
                or data_iter_step == len(data_loader) - 1
            )
        ):
            # Sync fid metric to ensure that the processes dont deviate much.
            # MG: wow, it syncs across processes
            gc.collect()
            running_fid = fid_metric.compute()
            shape_sampled = synthetic_samples.shape[0]
            logger.info(f"Evaluating: current batch size {shape_sampled},  [{num_synthetic}/{local_fid_samples}], running fid {running_fid}")

        if args.test_run:
            break
    
    metrics = {
        "fid": (
            float(fid_metric.compute().detach().cpu())
            if compute_fid_score
            else None
        ),
        "real_features": (
            torch.cat(real_feature_chunks, dim=0) if real_feature_chunks else None
        ),
        "fake_features": (
            torch.cat(fake_feature_chunks, dim=0) if fake_feature_chunks else None
        ),
    }
    return metrics
