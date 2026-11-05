#_Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path
import os

import torch
from training.distributed_mode import is_main_process

import logging


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def save_model(
    args, epoch_num, model_without_ddp, optimizer, lr_schedule, update_steps, mode,
):
    assert mode in ['last', 'periodic']

    ckpt_save_dir = Path(args.ckpt_save_dir)
    epoch_name = str(epoch_num)

    if mode == 'last':
        # this one keeps getting overwritten
        checkpoint_path = ckpt_save_dir / "checkpoint-last.pth"
        print("Will save the last.pth checkpoint at", checkpoint_path)
    else:
        assert mode == 'periodic'
        # this one is specific to a certain step
        checkpoint_path = ckpt_save_dir / ("checkpoint-step%s.pth" % update_steps)
        print("Will save a periodic checkpoint at", checkpoint_path)

    to_save = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_schedule": lr_schedule.state_dict() if lr_schedule is not None else None,
        "epoch_num": epoch_num,
        "args": args,
        'update_steps': update_steps,
    }
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(to_save, temporary_path)
    os.replace(temporary_path, checkpoint_path)
    step_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".step")
    temporary_step_path = step_path.with_suffix(step_path.suffix + ".tmp")
    temporary_step_path.write_text(f"{update_steps}\n")
    os.replace(temporary_step_path, step_path)
    #save_on_master(to_save, checkpoint_path)


def load_model(args, model_without_ddp, optimizer, lr_schedule, steps_per_epoch=None):

    update_steps = 0
    start_epoch = 0

    if not args.resume:
        return update_steps, start_epoch

    if args.resume.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.resume, map_location="cpu", check_hash=True
        )
    else:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
    logging.info("Resume checkpoint %s" % args.resume)
    
    
    if args.eval_only:
        logging.info("Loading ckpt but in eval mode, so skipping optimizer etc")
        # Still recover update_steps for logging/model_restored flag
        if 'update_steps' in checkpoint:
            update_steps = checkpoint['update_steps']
            logging.info(f"Eval mode: checkpoint was from step {update_steps}")
        # Sampling seeds include epoch_num, so evaluation must retain the
        # checkpoint's epoch rather than silently resetting it to zero.
        if 'epoch_num' in checkpoint:
            start_epoch = checkpoint['epoch_num']
            logging.info(f"Eval mode: using checkpoint epoch {start_epoch}")
    else:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info("Loaded Optimizer")
        else:
            logging.info("Failed to Load optimizer")

        
        if 'lr_schedule' in checkpoint:

            if lr_schedule is None:
                logging.info("Found LR schedule state dict in checkpoint, but cur lr_schedule is None, so not loading")
            else:
                lr_schedule.load_state_dict(checkpoint["lr_schedule"])
                logging.info("Loaded LR Schedule")
        else:
            logging.info("Failed to load LR Schedule")
            if lr_schedule is None:
                logging.info("And in any case, cur LR Schedule is Also None")
            else:
                logging.info("But cur lr schedule is not None. starting a new one")

        if 'update_steps' in checkpoint:
            update_steps = checkpoint['update_steps']
            logging.info(f"Recovered update_steps = {update_steps} from checkpoint")
        else:
            logging.info("No update_steps in checkpoint")

        # Derive start_epoch from update_steps
        if steps_per_epoch is not None and steps_per_epoch > 0:
            start_epoch = update_steps // steps_per_epoch
            logging.info(f"Resuming at epoch {start_epoch} (derived from step {update_steps})")
        else:
            # Fallback to saved epoch if steps_per_epoch not available
            for key in ('epoch_num', 'epoch'):
                if key in checkpoint:
                    start_epoch = checkpoint[key] + 1
                    logging.info(f"Loaded start_epoch={start_epoch} from checkpoint key '{key}'")
                    break

    return update_steps, start_epoch
