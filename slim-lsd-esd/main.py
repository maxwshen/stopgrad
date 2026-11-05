# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates.

import datetime
import logging
import os
import signal
import sys
import time
from pathlib import Path
import wandb
import argparse
import math

from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from models.model_configs import instantiate_model
from arg_parser import get_args_parser
import argparse
import gc
import logging
import math
from typing import Iterable, Any, Callable
import time
from models.time_sampler import sample_one_timestep, sample_two_timesteps


import torch
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.aggregation import MeanMetric
from training import distributed_mode as dm
from training.eval_loop import PreemptionRequested, create_fid_metric, eval_model
from training.load_and_save import load_model, save_model
from training.mind import compute_mind, gather_features
from torch.utils.tensorboard import SummaryWriter
from models.augment import AugmentPipe
import models.rng as rng
import utils
import data_utils


logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision('high')

def initialize(args):

    job_id = args.job_id
    task_id = args.task_id
    run_name = args.run_name
    
    base_results = Path(args.output_dir).expanduser().resolve() / run_name / str(job_id) / str(task_id)

    args.ckpt_save_dir = f"{base_results}/ckpts/"
    args.fid_dir = f"/tmp/{run_name}/{job_id}/{task_id}"
    args.wandb_dir = f"{base_results}/wandb/"

    dm.init_distributed_mode(args)
    print(f"Rank: {dm.get_rank()}")
    print(f"World Size: {dm.get_world_size()}")

    # MAKE DIRS
    if dm.is_main_process():
        if args.ckpt_save_dir:
            Path(args.ckpt_save_dir).mkdir(parents=True, exist_ok=True)
            print("created ckpt save dir:", args.ckpt_save_dir)
        if args.fid_dir:
            Path(args.fid_dir).mkdir(parents=True, exist_ok=True)
            print('created fid dir', args.fid_dir)
        if args.wandb_dir:
            Path(args.wandb_dir).mkdir(parents=True, exist_ok=True)
            print('created wandb dir', args.wandb_dir)

    # INIT WANDB
    if dm.get_rank() == 0:
        if args.use_wandb:
            config_dict = utils.namespace_to_dict(args)
            wandb_run_id = args.wandb_run_id or None
            wandb.init(                                             
                entity=args.wandb_entity,
                project=args.wandb_project,
                config=config_dict,
                id=wandb_run_id,
                name=args.run_name,
                resume="allow" if wandb_run_id else None,
                dir=args.wandb_dir,
            )

    # INIT LOGGER
    if dm.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        logger.addHandler(logging.NullHandler())

    logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    
    
    #logger.info("{}".format(args).replace(", ", ",\n"))
    
    if dm.is_main_process():
        # create tensorboard
        #os.makedirs(args.ckpt_save_dir, exist_ok=True) # done above.
        logger.info(f"Tensorboard writer created at {args.ckpt_save_dir}")
    else:
        logger.info('Writer not created.')

    # SET DEVICE AND SEEDS
    device = torch.device(args.device)
    # set the seeds
    seed = args.seed + dm.get_rank()  # legacy. TODO: rng.fold_in 
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # MAKE DATALOADERS
    logger.info(f"Initializing Dataset: {args.dataset}")
    data_loader_train = data_utils.get_data_loader(args, is_for_fid=False, logger=logger)
    data_loader_fid = data_utils.get_data_loader(args, is_for_fid=True, logger=logger)

    # define the model
    logger.info("Initializing Model")
    model = instantiate_model(args)

    model.to(device)
    model_without_ddp = model


    if args.print_model:
        utils.print_model(model, logger)

    eff_batch_size = args.batch_size * dm.get_world_size()

    logger.info(f"Learning rate: {args.lr:.2e}")

    logger.info(f"Effective batch size (bsz * world_size): {eff_batch_size}")

    # WRAP DDP
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu],
            find_unused_parameters=False,
            broadcast_buffers=False,
            static_graph=True,
            gradient_as_bucket_view=True
        )
        model_without_ddp = model.module

    return device, model, model_without_ddp, data_loader_train, data_loader_fid

class Trainer:

    def __init__(
        self,
        args,
        device,
        model,
        model_without_ddp,
        data_loader_train,
        data_loader_fid,

    ):
        self.args = args
        self.device = device
        self.model = model
        self.model_without_ddp = model_without_ddp
        self.data_loader_train = data_loader_train
        self.data_loader_fid = data_loader_fid

        self.augment_pipe = AugmentPipe(
            p=0.12, 
            xflip=1e8, 
            yflip=0, 
            scale=1, 
            rotate_frac=0, 
            aniso=1, 
            translate_frac=1
        )  # turn off yflip and rotate               

        self.setup_optimizer()
        self.setup_lr_schedule()
        self.possibly_load_ckpt()

        self.time_meter = MeanMetric().to(device, non_blocking=True)
        self.time_meter.reset()
        self.fid_metric = None
        self.fid_real_features_samples = None
        self.mind_real_features = None
        self.preemption_requested = False
        self.current_epoch_num = self.start_epoch
        signal.signal(signal.SIGUSR1, self._handle_preemption_signal)
        signal.signal(signal.SIGTERM, self._handle_preemption_signal)
        
        self.setup_compiled_forward_passes()

        self.overfit_samples, self.overfit_labels = next(iter(self.data_loader_train))
        logger.info(f"Start from {self.start_epoch} to {args.epochs} epochs")

        self.stop_training = False

    def _handle_preemption_signal(self, signum, frame):
        del frame
        self.preemption_requested = True
        logger.warning("Received signal %s; checkpointing at the next safe boundary.", signum)

    def save_preemption_checkpoint_and_exit(self):
        args = self.args
        logger.warning(
            "Saving emergency checkpoint at update step %s.", self.update_steps
        )
        if dm.is_main_process():
            save_model(
                args=args,
                model_without_ddp=self.model_without_ddp,
                optimizer=self.optimizer,
                lr_schedule=self.lr_schedule,
                epoch_num=self.current_epoch_num,
                update_steps=self.update_steps,
                mode="last",
            )
            sentinel = os.environ.get("STOPGRAD_PREEMPT_SENTINEL")
            if sentinel:
                Path(sentinel).parent.mkdir(parents=True, exist_ok=True)
                Path(sentinel).touch()
        if dm.is_dist_avail_and_initialized():
            dm.barrier()
        dm.cleanup()
        raise SystemExit(85)

    def setup_optimizer(self,):

        # SET OPTIMIZER
        self.optimizer = self.args.optimizer_cls(
            self.model_without_ddp.net.parameters(),
            lr=self.args.lr,
            betas=self.args.optimizer_betas,
            weight_decay=0.0
        )

    def setup_compiled_forward_passes(self):
        """Create lazy, phase-specific compiled callables.

        Eulerian and Lagrangian phase selection remains in eager Python so the
        compiled graphs never receive or branch on the global training step.
        The none/orig objectives need an Inductor workaround for an unsafe
        buffer reuse involving immutable ZeroTensors in the compiled backward.
        """
        disable = not self.args.compile
        second_order_compat = (
            self.args.compile
            and getattr(self.args, "compile_second_order", False)
            and self.args.which in ("eulerian", "lagrangian")
            and self.args.stopgrad_type in ("none", "orig")
        )
        offdiag_compile_kwargs = {}
        if second_order_compat:
            offdiag_compile_kwargs["options"] = {"allow_buffer_reuse": False}

        def forward_pass(model_without_ddp, *args, **kwargs):
            return model_without_ddp.forward_with_loss(*args, **kwargs)

        def velocity_forward_pass(model_without_ddp, *args, **kwargs):
            return model_without_ddp.velocity_loss(*args, **kwargs)

        def offdiag_forward_pass(model_without_ddp, *args, **kwargs):
            return model_without_ddp.offdiag_loss(*args, **kwargs)

        self.compiled_forward_pass = torch.compile(
            forward_pass,
            disable=disable,
        )
        self.compiled_velocity_forward_pass = torch.compile(
            velocity_forward_pass,
            disable=disable,
        )
        self.compiled_offdiag_forward_pass = torch.compile(
            offdiag_forward_pass,
            disable=disable,
            **offdiag_compile_kwargs,
        )
        self._compiled_phases_seen = set()

    def setup_lr_schedule(self,):
        self.lr_schedule = None

    def possibly_load_ckpt(self,):
  
        self.update_steps = 0
        self.start_epoch = 0

        # POSSIBLY LOAD CHECKPOINT
        steps_per_epoch = len(self.data_loader_train)
        self.update_steps, self.start_epoch = load_model(
            args=self.args,
            model_without_ddp=self.model_without_ddp,
            optimizer=self.optimizer,
            lr_schedule=self.lr_schedule, # could be None
            steps_per_epoch=steps_per_epoch,
        )
        self.model_restored = (self.update_steps > 0)
        self.resume_data_idx = (
            self.update_steps % steps_per_epoch if self.model_restored else 0
        )
        print("ran load_model")
        print("update steps is now", self.update_steps)
        print("start epoch is now", self.start_epoch)
        print("model restored is ", self.model_restored)
        print("resume data index is ", self.resume_data_idx)

    def maybe_save_ckpt(self, epoch_num):

        args = self.args

        if args.eval_only:
            return

        if not args.ckpt_save_dir:
            return 

        time_for_periodic = (self.update_steps % args.save_periodic_every == 0)
        time_for_most_recent = (self.update_steps % args.save_most_recent_every == 0)

        if (not time_for_periodic) and (not time_for_most_recent):
            return 

        common = {
            'args': args,
            'model_without_ddp': self.model_without_ddp,
            'optimizer': self.optimizer,
            'lr_schedule': self.lr_schedule, # could be None
            'epoch_num': epoch_num,
            'update_steps': self.update_steps,
        }

        if time_for_periodic:
            if dm.is_main_process():  
                save_model(**common, mode='periodic')
                logging.info(f"Saved checkpoint to {args.ckpt_save_dir}")
            dm.barrier()

        if time_for_most_recent:
            if dm.is_main_process():
                save_model(**common, mode='last')
                logging.info(f"Saved checkpoint to {args.ckpt_save_dir}")
            dm.barrier()

    def save_final_checkpoint(self):
        """Commit the exact terminal training state before reporting success."""
        args = self.args
        if args.eval_only or not args.ckpt_save_dir:
            return
        if dm.is_main_process():
            save_model(
                args=args,
                model_without_ddp=self.model_without_ddp,
                optimizer=self.optimizer,
                lr_schedule=self.lr_schedule,
                epoch_num=self.current_epoch_num,
                update_steps=self.update_steps,
                mode="last",
            )
            logging.info(
                "Saved final checkpoint at update step %s to %s",
                self.update_steps,
                args.ckpt_save_dir,
            )
        dm.barrier()
        if dm.is_main_process():
            sentinel = os.environ.get("STOPGRAD_COMPLETION_SENTINEL")
            if sentinel:
                sentinel_path = Path(sentinel)
                sentinel_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path = sentinel_path.with_suffix(
                    sentinel_path.suffix + ".tmp"
                )
                temporary_path.write_text("")
                os.replace(temporary_path, sentinel_path)
                logging.info("Committed completion marker %s", sentinel_path)
        dm.barrier()

    def maybe_sample(self, epoch_num, log_dict=None):

        args =  self.args

        if log_dict is None:
            log_dict = {}
      
        # these samples only go to wandb so no need to do if no wandb
        ready = (self.update_steps % args.sample_every == 0) and args.use_wandb
        if not ready:
            return log_dict
        if not dm.is_main_process():
            return log_dict
         
        logging.info("Sampling During Training")
        for nfe in self.args.sample_which_steps:
            for use_ema in [False, True]:
                if self.preemption_requested:
                    raise PreemptionRequested
                log_dict = self.sample(
                    log_dict = log_dict,
                    nfe = nfe,
                    use_ema = use_ema,
                )

        return log_dict

    @torch.no_grad()
    def sample(self, nfe=1, use_ema=False, log_dict=None):

        args = self.args

        if log_dict is None:
            log_dict = {}

        with torch.random.fork_rng(devices=[self.device]):
            torch.manual_seed(rng.fold_in(args.seed, rng.get_rank(), 0, self.update_steps))
            with torch.amp.autocast('cuda', enabled=False), torch.no_grad():
                
                if use_ema:
                    net = self.model_without_ddp.net_ema
                else:
                    net = self.model_without_ddp.net
                self.eval_mode()

                samples = self.model_without_ddp.sample(
                    samples_shape = (args.batch_size_sampling_plotting, 3, 32, 32),
                    net = net,
                    device = self.device,
                    nfe = nfe,
                )
        torch.cuda.synchronize()
        ema_str = 'ema_' if use_ema else 'nonema_'
        key = ema_str + 'samples_' + str(nfe)
        log_dict[key] = utils.wrap_wandb(samples)
        return log_dict


    def maybe_compute_fid(self, epoch_num, log_dict=None):
       
        args = self.args

        if log_dict is None:
            log_dict = {}

        if not args.compute_fid:
            return log_dict

        time_to_fid = (self.update_steps % args.fid_every == 0)
        not_too_soon = (self.update_steps >= args.compute_fid_after)
        ready = args.compute_fid and time_to_fid and not_too_soon
        
        if not ready:
            return log_dict

        logging.info("Evaluating FID")

        # these samples only go to wandb so no need to do if no wandb
        ema_options = [True] if args.ema_only_for_fid else [False, True]
        for nfe in args.nfes_for_fid:
            for use_ema in ema_options:
                log_dict=self.compute_fid(
                    epoch_num=epoch_num,
                    log_dict=log_dict,
                    use_ema = use_ema,
                    nfe = nfe,
                    fid_samples = args.fid_samples_during_training,
                )
        return log_dict

    def maybe_compute_mind(self, epoch_num, log_dict=None):
        args = self.args

        if log_dict is None:
            log_dict = {}

        if not args.compute_mind:
            return log_dict
        if self.update_steps == 0 or self.update_steps % args.mind_every != 0:
            return log_dict

        logging.info("Evaluating MIND")
        if self.fid_metric is None:
            self.fid_metric = create_fid_metric(self.device)

        local_mind_samples = math.ceil(
            args.mind_samples / dm.get_world_size()
        )
        reference_features = None
        for model_name, net_eval in (
            ("nonema", self.model_without_ddp.net),
            ("ema", self.model_without_ddp.net_ema),
        ):
            compute_real_features = self.mind_real_features is None
            eval_stats = eval_model(
                model=self.model,
                net_eval=net_eval,
                nfe=args.mind_nfe,
                data_loader=self.data_loader_fid,
                device=self.device,
                epoch_num=epoch_num,
                update_steps=self.update_steps,
                args=args,
                fid_samples=args.mind_samples,
                overfit_samples=self.overfit_samples,
                overfit_labels=self.overfit_labels,
                fid_metric=self.fid_metric,
                compute_real_features=compute_real_features,
                feature_samples=local_mind_samples,
                collect_real_features=compute_real_features,
                collect_fake_features=True,
                compute_fid_score=False,
                should_stop=lambda: self.preemption_requested,
            )
            if eval_stats["real_features"] is not None:
                self.mind_real_features = eval_stats["real_features"]
            if (
                self.mind_real_features is None
                or eval_stats["fake_features"] is None
            ):
                raise RuntimeError("MIND feature collection was incomplete")

            if reference_features is None:
                reference_features = gather_features(
                    self.mind_real_features,
                    args.mind_samples,
                )
            generated_features = gather_features(
                eval_stats["fake_features"],
                args.mind_samples,
            )
            mind_score = compute_mind(
                generated_features,
                reference_features,
                rng_seed=args.mind_rng_seed,
                n_projections=args.mind_n_projections,
            )
            mind_key = (
                f"eval/MIND-{args.mind_samples // 1000}k-"
                f"{args.mind_nfe}step-{model_name}"
            )
            log_dict[mind_key] = float(mind_score.detach().cpu())
            logging.info(
                "Update Steps:%s. %s: %s",
                self.update_steps,
                mind_key,
                log_dict[mind_key],
            )
        return log_dict

    def compute_fid(
        self,
        epoch_num,
        log_dict=None,
        use_ema=False,
        nfe=1,
        fid_samples=2,
        cache_real_features=True,
    ):

        if log_dict is None:
            log_dict = {}

        if not self.args.compute_fid:
            return log_dict

        if use_ema:
            net_eval = self.model_without_ddp.net_ema 
            ema_str = 'ema'
        else:
            net_eval = self.model_without_ddp.net
            ema_str = 'nonema'

        if cache_real_features:
            if (
                self.fid_metric is None
                or self.fid_real_features_samples != fid_samples
            ):
                self.fid_metric = create_fid_metric(self.device)
                self.fid_real_features_samples = None
            fid_metric = self.fid_metric
            compute_real_features = self.fid_real_features_samples is None
        else:
            # The startup smoke test uses only a few samples and must not
            # populate the persistent real-feature cache.
            fid_metric = create_fid_metric(self.device)
            compute_real_features = True

        eval_stats = eval_model(
            model=self.model, 
            net_eval=net_eval, 
            nfe=nfe, 
            data_loader=self.data_loader_fid, 
            device=self.device, 
            epoch_num=epoch_num, 
            update_steps = self.update_steps,
            args=self.args, 
            fid_samples = fid_samples,
            overfit_samples=self.overfit_samples,
            overfit_labels=self.overfit_labels,
            fid_metric=fid_metric,
            compute_real_features=compute_real_features,
            should_stop=lambda: self.preemption_requested,
        )
        if cache_real_features and compute_real_features:
            self.fid_real_features_samples = fid_samples
        fid_score = eval_stats['fid']
        logging.info(f"Update Steps:{self.update_steps}. FID {ema_str} nfe={nfe}: {fid_score}")
        key = ema_str + '_fid_' + str(nfe)
        log_dict[key] = fid_score
        return log_dict


    def maybe_log(self, epoch_num, data_idx, log_dict):
      
        if (self.update_steps % self.args.wandb_every == 0):

            log_dict['sec_per_iter'] = self.time_meter.compute()
            self.time_meter.reset()

            log_dict['epoch_num'] = epoch_num
            log_dict['update_steps'] = self.update_steps

            # plot to wandb on rank0
            if dm.is_main_process():
                if self.args.use_wandb:
                    logging.info("Log Wandb")
                    wandb.log(log_dict, step=self.update_steps)
            all_str = f"Epoch {epoch_num} [{data_idx}/{len(self.data_loader_train)}]:"
            logger.info(all_str)



    def prepare_batch(self, x):
        x = x.to(self.device, non_blocking=True)
        x = x * 2.0 - 1.0 # center
        if self.args.use_edm_aug:
            seed = rng.fold_in(self.args.seed, self.update_steps, rng.get_rank(), "augment")
            with torch.random.fork_rng(devices=[x.device], enabled=True):
                torch.manual_seed(seed)
                if x.is_cuda:
                    torch.cuda.manual_seed(seed)
                x, aug_cond = self.augment_pipe(x)
        else:
            aug_cond = None
        return x, aug_cond


    def get_one_time(self, batch_size):
        t = sample_one_timestep(
            self.args,
            batch_size=batch_size,
            device=self.device
        )
        return t

    def get_two_times(self, batch_size):
        t, u = sample_two_timesteps(
            self.args, 
            batch_size=batch_size, 
            device=self.device
        )
        return t, u

    def still_warmup(self):
        do_warmup = self.args.do_velocity_warmup
        still = (self.update_steps < self.args.velocity_warmup_steps)
        return do_warmup and still

    def get_loss_args(self, x):
        x, aug_cond = self.prepare_batch(x)
        noise = torch.randn_like(x).type_as(x)

        if self.still_warmup() or (self.args.which == 'flowmatching'):
            t = self.get_one_time(batch_size=x.shape[0])
            u = None
        else:
            t, u = self.get_two_times(batch_size=x.shape[0])
        return x, aug_cond, t, u, noise

    def _train_step_no_sync(self, x):

        self.optimizer.zero_grad(set_to_none=True)

        x, aug_cond, t, u, noise = self.get_loss_args(x)

        seed = rng.fold_in(self.args.seed, self.update_steps, rng.get_rank(), "train_step")
        with torch.random.fork_rng(devices=[x.device], enabled=True):
            torch.manual_seed(seed)
            if x.is_cuda:
                torch.cuda.manual_seed(seed)

            if self.args.which in ("eulerian", "lagrangian"):
                if self.still_warmup():
                    phase = "velocity warmup"
                    forward = self.compiled_velocity_forward_pass
                    forward_kwargs = {
                        "x": x,
                        "aug_cond": aug_cond,
                        "noise": noise,
                        "t": t,
                    }
                else:
                    phase = "full off-diagonal"
                    forward = self.compiled_offdiag_forward_pass
                    forward_kwargs = {
                        "x": x,
                        "aug_cond": aug_cond,
                        "noise": noise,
                        "t": t,
                        "u": u,
                    }
            else:
                phase = "training"
                forward = self.compiled_forward_pass
                forward_kwargs = {
                    "x": x,
                    "aug_cond": aug_cond,
                    "t": t,
                    "u": u,
                    "noise": noise,
                    "step_for_loss_fn": self.update_steps,
                }

            if self.args.compile and phase not in self._compiled_phases_seen:
                logging.info(
                    "Compiling the %s graph; this may take a while...",
                    phase,
                )
                self._compiled_phases_seen.add(phase)
            loss = forward(self.model_without_ddp, **forward_kwargs)

        if self.args.compile:
            assert utils.get_compiled_counts() > 0, "Compilation not triggered."

        if self.args.debug_checks and not torch.isfinite(loss).all():
            raise ValueError(f"Loss is {loss}, stopping training")

        loss.backward(create_graph=False)

        return loss.detach()

    def train_step(self, x, data_idx, epoch_num, log_dict=None):

        if log_dict is None:
            log_dict = {}

        loss = self._train_step_no_sync(x)

        # To support compiling, we need to call model.module and then sync gradients.
        utils.synchronize_gradients(
            self.model,
            bucket_cap_mb=self.args.gradient_bucket_cap_mb,
        )
       
        max_norm = self.args.grad_clip_norm
        if max_norm is not None and max_norm >= 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model_without_ddp.parameters(), max_norm
            )
        else:
            grad_norm = -1


        # check the grads
        if (
            self.args.debug_checks
            and (epoch_num - self.start_epoch) % 100 == 0
            and data_idx < 2
        ):
            logging.info(f"Doing Sanity Check, step is {self.update_steps}")
            utils.gradient_sanity_check(self.model)

        # update the parameters
        self.optimizer.step()

        # update the LR
        new_lr = self.get_lr()
        for pg in self.optimizer.param_groups:
            old_lr = pg['lr']
            pg["lr"] = new_lr

        # update the EMA
        self.model_without_ddp.update_ema(step = self.update_steps)

        # update the step counter
        self.update_steps += 1

        log_dict['loss'] = loss
        log_dict['grad_norm'] = grad_norm
        log_dict['lr'] = new_lr
        return log_dict

    def get_lr(self):
        schedule = self.args.lr_schedule
        lr_decay_steps = None
        step = self.update_steps
        base_lr = self.args.lr
        min_lr = self.args.min_lr
        num_training_steps = self.args.epochs * len(self.data_loader_train)
        lr_warmup_steps = self.args.lr_warmup_steps
        is_resumed = self.model_restored        

        if schedule == "constant":
            lr_no_warmup = base_lr
        elif schedule == "cosine":
            rel_step = max(0, step - lr_warmup_steps)
            rel_total = max(1, num_training_steps - lr_warmup_steps)
            ratio = rel_step / rel_total
            lr_no_warmup = 0.5 * (1 + math.cos(math.pi * ratio)) * base_lr
        elif schedule == 'linear': 
            multiplier = .992 ** (step // 1000)
            lr_no_warmup = multiplier * base_lr
        elif schedule == 'sqrt':
            numer = base_lr 
            denom = np.sqrt(max(step / lr_decay_steps, 1.0))
            lr_no_warmup = numer / denom
        else:
            assert False

        if is_resumed or (lr_warmup_steps < 1):
            warmup_coef = 1.0
        else:
            warmup_coef = step / lr_warmup_steps if step < lr_warmup_steps else 1.0
        lr = warmup_coef * lr_no_warmup
        return max(lr, min_lr)

    def train_mode(self,):
        self.model.train(True)
        self.model_without_ddp.train(True)
        self.model.train()
        self.model_without_ddp.train()

    def eval_mode(self,):
        self.model.train(False)
        self.model_without_ddp.train(False)
        self.model.eval()
        self.model_without_ddp.eval()

    def train_epoch(self, epoch_num):
        
        args = self.args
        self.current_epoch_num = epoch_num

        gc.collect()
        
        # SET RANDOMNESS IN DISTRIBUTED DATALOADER FOR THIS EPOCH
        if args.distributed:
            self.data_loader_train.sampler.set_epoch(epoch_num)

        for data_idx, (samples, labels) in enumerate(self.data_loader_train):

            log_dict = {}

            if epoch_num == self.start_epoch and data_idx < self.resume_data_idx:
                continue

            if data_idx > 0 and args.test_run:
                break

            # train on one batch of samples for quick visualization
            if args.overfit:
                samples = self.overfit_samples
                labels = self.overfit_labels

            if not args.eval_only:
        
                # switch into train mode
                self.train_mode()

                # take timed train step
                tic = time.time()
                log_dict = self.train_step(
                    x = samples,
                    data_idx = data_idx,
                    epoch_num = epoch_num,
                    log_dict = log_dict,
                )
                toc = time.time()
                
                # update train meters
                self.time_meter.update(toc - tic)
                if self.preemption_requested:
                    raise PreemptionRequested

            # checkpoint
            self.maybe_save_ckpt(epoch_num)
            
            # sampling (skip in eval_only mode)
            if not args.eval_only:
                log_dict = self.maybe_sample(
                    epoch_num = epoch_num,
                    log_dict = log_dict,
                )
            
            # fid
            log_dict = self.maybe_compute_mind(
                epoch_num=epoch_num,
                log_dict=log_dict,
            )

            log_dict = self.maybe_compute_fid(
                epoch_num = epoch_num,
                log_dict = log_dict,
            )
            
            # wandb
            self.maybe_log(
                epoch_num = epoch_num,
                data_idx = data_idx,
                log_dict = log_dict
            )

            if args.max_train_steps > 0 and self.update_steps >= args.max_train_steps:
                logger.info(f"Reached max_train_steps={args.max_train_steps} at update_steps={self.update_steps}; stopping.")
                self.stop_training = True
                break

            if args.test_run or args.eval_only:
                break


        logging.info(f"Done with Epoch {epoch_num}")

    def train_loop(self,):

        self.start_time = time.time()

        try:
            if (
                self.args.max_train_steps > 0
                and self.update_steps >= self.args.max_train_steps
            ):
                logger.info(
                    "Checkpoint is already at update_steps=%s "
                    "(max_train_steps=%s); no additional training is needed.",
                    self.update_steps,
                    self.args.max_train_steps,
                )
                self.stop_training = True

            if (
                not self.stop_training
                and self.args.compute_fid
                and not self.args.eval_only
            ):
                # compute QUICK FID to make sure inception model is all set
                # to avoid an annoying crash in a long training run
                # (skip in eval_only mode since real FID runs immediately)
                logging.info('Doing quick FID to ensure Inception Model Available')
                _ = self.compute_fid(
                    epoch_num = 0,
                    log_dict = None,
                    use_ema = True,
                    nfe = 1,
                    fid_samples = 5,
                    cache_real_features = False,
                )
                logging.info("Done with quick FID test")

            # START TRAIN LOOP
            for epoch_num in range(self.start_epoch, self.args.epochs):
                if self.stop_training:
                    break

                self.train_epoch(epoch_num)

                if self.stop_training:
                    break

                if self.args.test_run or self.args.eval_only:
                    break
        except PreemptionRequested:
            self.save_preemption_checkpoint_and_exit()
        except Exception:
            # Slurm terminates every process in the job step, including
            # DataLoader workers.  Their death can surface as a RuntimeError
            # before train_step() returns to the normal safe-boundary check.
            # Only reinterpret such an exception when this rank actually
            # observed a preemption signal; ordinary training failures must
            # still propagate.
            if not self.preemption_requested:
                raise
            logger.exception(
                "Training was interrupted after a preemption signal; "
                "checkpointing the last fully completed optimizer step."
            )
            self.save_preemption_checkpoint_and_exit()

        self.save_final_checkpoint()
        total_time = time.time() - self.start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(f"Training time {total_time_str}")
        dm.cleanup()



         
if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    args.optimizer_cls = torch.optim.AdamW

    args.do_velocity_warmup = (args.velocity_warmup_steps >= 1)

    if args.do_velocity_warmup:
        args.update_ema_after = max(args.update_ema_after, args.velocity_warmup_steps)

    args.overfit = bool(args.overfit) # 0/1 to False/True
    args.test_run = bool(args.test_run)
    args.compile = bool(args.compile)
    args.compile_second_order = bool(args.compile_second_order)
    args.debug_checks = bool(args.debug_checks)
    args.eval_only = bool(args.eval_only)
    args.compute_fid = bool(args.compute_fid)
    args.compute_mind = bool(args.compute_mind)
    args.pin_mem = bool(args.pin_mem)
    args.use_wandb = bool(args.use_wandb)
    args.print_model = bool(args.print_model)
    args.adaptive_weight_flowmap = bool(args.adaptive_weight_flowmap)
    args.use_edm_aug = bool(args.use_edm_aug)

    if args.overfit:
        args.compute_fid = False
        args.save_most_recent_every = 1000000000
        args.save_periodic_every = 1000000000
        args.fid_every = 100000000
        args.wandb_every = 100
        args.sample_every = 500
        args.lr_warmup_steps = 0
        args.dropout = 0.0
        #args.debug_sampler = True

    if args.test_run:
        args.overfit = False
        args.fid_every = 1
        args.compute_fid_after = 0
        args.fid_samples_during_training = 2
        args.update_ema_after = 0
        args.update_ema_period = 1
        args.save_most_recent_every = 1
        args.save_periodic_every = 1
        args.sample_every = 1
        args.wandb_every = 1


    device, model, model_without_ddp, data_loader_train, data_loader_fid = initialize(args)
    trainer = Trainer(
        args=args,
        device=device,
        model=model,
        model_without_ddp = model_without_ddp,
        data_loader_train=data_loader_train,
        data_loader_fid=data_loader_fid,
    )
    trainer.train_loop()
