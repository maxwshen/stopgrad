# Slim ESD and LSD

CIFAR-10 training code for slim Eulerian self-distillation (ESD) and slim
Lagrangian self-distillation (LSD).

Run the commands below from this directory:

```bash
cd slim-lsd-esd
```

## Installation

Use Python 3.12 on Linux with an NVIDIA GPU. For a new environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install CUDA-enabled PyTorch and matching torchvision using the
[PyTorch installation instructions](https://pytorch.org/get-started/locally/),
then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The reference environment uses PyTorch 2.13.0+cu130, torchvision 0.28.0+cu130,
torchmetrics 1.9.0, and torch-fidelity 0.4.0. Requirements are unpinned so that
you can choose a PyTorch build for your driver. Compiler behavior for
higher-order derivatives can vary across versions.

The training script uses the active environment. You can also set
`VENV_PATH=/path/to/venv` to activate one, or `PYTHON_BIN=/path/to/python` to
choose the interpreter.

## Training

For slim ESD on one GPU:

```bash
WHICH=eulerian STOPGRAD_MODE=simp SEED=0 GPUS=1 bash scripts/train.sh
```

For slim LSD, set `WHICH=lagrangian`. The code calls the slim objectives `simp`.
The available methods are:

| Method | `WHICH` | `STOPGRAD_MODE` |
| --- | --- | --- |
| Slim ESD | `eulerian` | `simp` |
| Slim LSD | `lagrangian` | `simp` |
| Original stopgrad ESD/LSD | `eulerian` / `lagrangian` | `orig` |
| ESD/LSD without stopgrad | `eulerian` / `lagrangian` | `none` |
| MeanFlow | `meanflow` | Ignored |
| Flow matching | `flowmatching` | Ignored |

MeanFlow uses its own fixed stopgrad objective. Improved MeanFlow (iMF) is
not implemented.

The launcher uses batch size 128 per GPU, 40,000 velocity-only warmup steps,
and 200,100 total optimizer steps. Compilation is enabled. EMA starts at step
40,000 with decay 0.9999. FID uses 50,000 samples every 25,000 steps, starting
after step 24,000, with both EMA and non-EMA models at NFE 1/10/50/100.
W&B and MIND are off by default. MeanFlow and flow matching skip velocity warmup.

Set `GPUS=4` for four GPUs (global batch size 512 at the default batch size).
Other launcher settings are `BATCH_SIZE`, `VELOCITY_WARMUP_STEPS`, `SEED`,
`COMPILE`, `DATA_ROOT`, `OUTPUT_ROOT`, and `RUN_NAME`. Setting
`VELOCITY_WARMUP_STEPS=0` trains ESD/LSD from scratch without velocity warmup.
Extra arguments are passed to `main.py`; for example, a short run without FID:

```bash
WHICH=lagrangian STOPGRAD_MODE=simp GPUS=1 BATCH_SIZE=2 COMPILE=0 \
  bash scripts/train.sh --max_train_steps 2 --compute_fid 0
```

Use the environment variables to set the method, paths, seed and run name.
The short run still loads the full model. `python main.py --help` lists the
remaining options. Direct `main.py` invocation defaults to MeanFlow, batch size
64, and compilation off, so use the launcher for the settings above.

CIFAR-10 downloads automatically to `DATA_ROOT` (default: `./data`).
`--dataset mnist` is also available for development; images are converted to
three-channel 32-by-32 inputs. ImageNet and latent-space training are not included.

## Compilation and memory

The launcher compiles all ESD/LSD variants. For `none` and `orig`, it also sets
`--compile_second_order 1`, which disables Inductor buffer reuse for the
off-diagonal graph. Velocity warmup and the full objective have separate
compiled functions.

The [compiled B200 benchmark](benchmarks/README.md) measures about 2x lower
peak allocated GPU memory for slim ESD/LSD than for the objectives without
stopgrad. The benchmark includes step timings and measurement conditions.

## Checkpoints

Runs write to `OUTPUT_ROOT/RUN_NAME/JOB_ID/SEED/`, where `OUTPUT_ROOT` defaults
to `./outputs` and `JOB_ID` is 0 outside Slurm. Checkpoints are in `ckpts/`:

- `checkpoint-last.pth` is saved every 5,000 steps and on completion.
- `checkpoint-stepNNNN.pth` is saved every 10,000 steps.
- `*.pth.step` files record the step for checkpoint selection.

The launcher resumes the newest checkpoint in the run directory. If that
directory has none, `RESUME_CHECKPOINT=/path/to/checkpoint.pth` supplies one.
For direct invocation, use `--resume`. A `training-complete` marker causes the
launcher to skip a finished run; choose a new `RUN_NAME` to start fresh.

Resume restores the model, EMA, optimizer and step counters. It does not save
the full RNG state, so continuation need not be bit-identical.

## Slurm

Submit from `slim-lsd-esd/`. The template requests one node, four GPUs and two
days; set the account, partition and resources for your cluster:

```bash
WHICH=eulerian STOPGRAD_MODE=simp SEED=0 \
  sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION scripts/train.sbatch
```

Activate your environment or export `VENV_PATH` before submission. If you
change the GPU allocation, also set `GPUS` to the number allocated. When
submitting from another directory, set `REPO_ROOT` to the absolute path of
`slim-lsd-esd/`.

To run all six objectives with three seeds:

```bash
for method in eulerian lagrangian; do
  for variant in none orig simp; do
    for seed in 0 1 2; do
      WHICH="$method" STOPGRAD_MODE="$variant" SEED="$seed" \
        sbatch scripts/train.sbatch
    done
  done
done
```

The wrapper runs `torchrun` in one `srun` step. On preemption, workers attempt
an emergency checkpoint. The job requeues after the save completes. A requeued job requires a checkpoint.

## Evaluation

Evaluate a slim-ESD checkpoint with EMA:

```bash
python main.py \
  --which eulerian --stopgrad_type simp \
  --resume /path/to/checkpoint.pth --eval_only 1 \
  --data_path ./data --output_dir ./outputs --run_name evaluation \
  --compute_fid 1 --compute_fid_after 0 --fid_every 1 \
  --ema_only_for_fid 1 --nfes_for_fid 1 10 50 100 \
  --use_wandb 0
```

Use the checkpoint's method and architecture. FID uses the training split as
its real-image reference and downloads Inception weights on first use.
Results go to the terminal and, if enabled, W&B. `--compute_mind 1` enables
the optional MIND metric.

## Benchmarking

Run all six objectives with compilation on one GPU:

```bash
python scripts/benchmark_training_speed.py \
  --compile --compile-second-order --batch-size 128 \
  --warmup-steps 2 --steps 5 --output scripts/benchmark_results/compiled.csv
```

This measures synchronized optimizer steps on synthetic CIFAR-shaped inputs,
including forward, backward, optimizer and EMA updates. It excludes data
loading, evaluation and checkpointing. The script writes CSV, Markdown and
JSON files with timings, peak allocated memory and run metadata. Use more
warmup and measured steps for longer runs, and check that compilation has
finished before interpreting timings.

## W&B

Logging is off by default. To enable it:

```bash
wandb login
WANDB_ENTITY=YOUR_ENTITY WANDB_PROJECT=stopgrad USE_WANDB=1 \
  WHICH=eulerian STOPGRAD_MODE=simp bash scripts/train.sh
```

The launcher sets a stable run ID for resume. Set `WANDB_RUN_ID` to choose one.

## Tests

Launcher and checkpoint-selection tests run without a GPU or Slurm:

```bash
python -m unittest discover -s tests -v
```

## Acknowledgments

This code builds on [MeanFlow](https://github.com/Gsunshine/py-meanflow),
[flow-map matching](https://github.com/nmboffi/flow-maps), and the implementations
credited in source-file headers. Their copyright notices are retained.
