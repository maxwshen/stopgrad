# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import logging
import os


logger = logging.getLogger(__name__)


def get_args_parser():
    parser = argparse.ArgumentParser("Image dataset training")


    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY"),
        help="W&B entity. Defaults to the WANDB_ENTITY environment variable.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=os.environ.get("WANDB_PROJECT"),
        help="W&B project. Defaults to the WANDB_PROJECT environment variable.",
    )
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        default=os.environ.get("WANDB_RUN_ID", ""),
        help="Stable W&B run ID. Defaults to WANDB_RUN_ID.",
    )

    parser.add_argument('--run_name', type = str, default = 'debug')
    parser.add_argument('--overfit', default = 0, type = int)
    parser.add_argument("--test_run", type = int, default = 0)
    parser.add_argument("--compile", type = int, default = 0)
    parser.add_argument(
        "--compile_second_order",
        type=int,
        default=0,
        help=(
            "Enable the Inductor compatibility path needed to compile the "
            "second-order none/orig objectives. This disables unrelated "
            "buffer reuse only for the compiled off-diagonal graph."
        ),
    )
    parser.add_argument(
        "--debug_checks",
        type=int,
        default=0,
        help="Enable synchronous finite-loss and distributed gradient sanity checks.",
    )
    parser.add_argument("--eval_only", type = int, default = 0)
    parser.add_argument("--compute_fid", type = int, default = 1)
    parser.add_argument(
        "--compute_mind",
        type=int,
        default=0,
        help="Compute MIND on non-EMA and EMA samples at the configured interval.",
    )
    parser.add_argument("--pin_mem", type = int, default = 1)
    parser.add_argument('--use_wandb', type = int, default = 0)
    parser.add_argument('--print_model', type = int, default = 0)

    parser.add_argument('--job_id', type = int, default = 0)
    parser.add_argument('--task_id', type = int, default = 0)

    # Optimizer parameters
    parser.add_argument('--compute_fid_after', type = int, default = 24000)
    parser.add_argument('--nfes_for_fid', type=int, nargs='+', default=[1, 10, 50, 100])
    parser.add_argument('--ema_only_for_fid', type=int, default=0, help="If 1, only evaluate EMA model for FID")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size per GPU")
    parser.add_argument("--batch_size_sampling", default=32, type=int, help="Batch size per GPU")
    parser.add_argument("--epochs", default=16000, type=int)
    parser.add_argument("--max_train_steps", default=0, type=int, help="If > 0, stop training once update_steps >= this value (checked at end of iteration, after wandb log).")
    parser.add_argument("--optimizer_betas", default=[0.9, 0.999], nargs="+", type=float, help="adam betas")
    parser.add_argument('--which', default='meanflow', choices=['flowmatching', 'meanflow', 'lagrangian', 'eulerian'])
    parser.add_argument('--stopgrad_type', type=str, default='none', choices=['none', 'orig', 'simp'])
    parser.add_argument('--arch_name', type=str, default='song_unet', choices=['song_unet', 'mg_unet', 'edm2_unet'])
    
    # Dataset parameters
    parser.add_argument("--dataset", default='cifar10', type=str, choices=['cifar10', 'mnist'], help="Dataset to use.")
    parser.add_argument("--data_path", default="./data", type=str, help="data root folder with train, val and test subfolders")
    parser.add_argument("--fid_samples_during_training", default=50000, type=int, help="number of synthetic samples for FID evaluations")
    parser.add_argument("--mind_samples", default=5000, type=int)
    parser.add_argument("--mind_n_projections", default=100, type=int)
    parser.add_argument("--mind_rng_seed", default=0, type=int)
    parser.add_argument("--mind_nfe", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch (used when resumed from checkpoint)")
    parser.add_argument('--save_most_recent_every', default = 5000, type = int)
    parser.add_argument('--save_periodic_every', default = 10000, type = int)
    parser.add_argument('--sample_every', default = 10000, type = int)
    parser.add_argument("--fid_every", default=25000, type=int)
    parser.add_argument("--mind_every", default=5000, type=int)
    parser.add_argument('--wandb_every', default = 100, type = int)

    #parser.add_argument("--save_fid_samples", action="store_true", help="Save all samples generated for FID computation.")
    parser.add_argument("--num_workers", default=4, type=int)

    # Distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    parser.add_argument(
        "--gradient_bucket_cap_mb",
        default=25,
        type=float,
        help="Maximum size in MiB for each manual gradient all-reduce bucket.",
    )

    # MeanFlow specific parameters
    parser.add_argument("--prob_t_eq_u", default=0.25, type=float, help="Probability of sampling t equal to u")
    parser.add_argument("--adaptive_weight_flowmap", default=1, type=int)
    parser.add_argument("--velocity_warmup_steps", default=0, type=int, help="If >= 1, do velocity-only loss first ")
    parser.add_argument("--norm_p", default=0.75, type=float, help="Norm power for adaptive weight.")
    parser.add_argument("--norm_eps", default=1e-3, type=float, help="Small constant for adaptive weight division.")
    parser.add_argument(
        "--output_dir",
        default="./outputs",
        type=str,
        help="Output root; runs use RUN_NAME/JOB_ID/TASK_ID beneath this directory.",
    )

    # Training hyperparameters (previously hardcoded in main.py)
    parser.add_argument("--grad_clip_norm", default=1.0, type=float, help="Max gradient norm for clipping")
    parser.add_argument("--lr_warmup_steps", default=20000, type=int, help="Number of LR warmup steps")
    parser.add_argument("--lr", default=0.0002, type=float, help="Learning rate")
    parser.add_argument("--lr_schedule", default="constant", type=str, choices=["constant", "cosine", "linear", "sqrt"], help="LR schedule type")
    parser.add_argument("--min_lr", default=1e-6, type=float, help="Minimum learning rate")
    parser.add_argument("--dropout", default=0.20, type=float, help="Dropout rate")

    # Time sampling bounds
    parser.add_argument("--time_min_sampling", default=1e-4, type=float, help="Min time for sampling")
    parser.add_argument("--time_max_sampling", default=0.9999, type=float, help="Max time for sampling")
    parser.add_argument("--time_min_training", default=1e-4, type=float, help="Min time for training")
    parser.add_argument("--time_max_training", default=0.9999, type=float, help="Max time for training")

    # EMA
    parser.add_argument("--ema_decay", default=0.9999, type=float, help="EMA decay rate")
    parser.add_argument("--update_ema_after", default=40000, type=int, help="Start EMA updates after this many steps")
    parser.add_argument("--update_ema_period", default=1, type=int, help="EMA update period in steps")

    # Augmentation
    parser.add_argument("--use_edm_aug", default=0, type=int, help="Use EDM augmentation pipeline")

    # Sampling
    parser.add_argument("--sample_which_steps", default=[1, 2, 10, 50, 100], nargs="+", type=int, help="NFE values to sample at during training")
    parser.add_argument("--batch_size_sampling_fid", default=128, type=int, help="Batch size for FID sampling")
    parser.add_argument("--batch_size_sampling_plotting", default=64, type=int, help="Batch size for plot sampling")

    return parser
