#!/usr/bin/env python3
"""Benchmark full off-diagonal training steps on one local CUDA device.

Run from the slim-lsd-esd directory:

    .venv/bin/python scripts/benchmark_training_speed.py

The benchmark uses the production Trainer.train_step implementation but omits
data loading, W&B, evaluation, sampling, and checkpointing. Synthetic
CIFAR-shaped inputs keep the comparison focused on training compute.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import subprocess
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arg_parser import get_args_parser  # noqa: E402
from main import Trainer  # noqa: E402
from models.augment import AugmentPipe  # noqa: E402
from models.model_configs import instantiate_model  # noqa: E402


SETUPS = tuple(
    (method, stopgrad)
    for method in ("eulerian", "lagrangian")
    for stopgrad in ("none", "orig", "simp")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare single-GPU training-step speed for all stopgrad setups."
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Benchmark torch.compile; compilation time is excluded by warmup steps.",
    )
    parser.add_argument(
        "--compile-second-order",
        action="store_true",
        help=(
            "Enable the opt-in Inductor compatibility path for compiled "
            "none/orig objectives."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path. Defaults to a timestamped file under benchmark_results/.",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.warmup_steps < 1:
        parser.error("--warmup-steps must be at least 1")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    return args


def training_args(
    benchmark_args: argparse.Namespace,
    method: str,
    stopgrad: str,
) -> argparse.Namespace:
    args = get_args_parser().parse_args([])
    args.which = method
    args.stopgrad_type = stopgrad
    args.batch_size = benchmark_args.batch_size
    args.seed = benchmark_args.seed
    args.device = benchmark_args.device
    args.compile = benchmark_args.compile
    args.compile_second_order = benchmark_args.compile_second_order
    args.debug_checks = False
    args.optimizer_cls = torch.optim.AdamW

    # Force the requested full, strictly off-diagonal training path.
    args.velocity_warmup_steps = 0
    args.do_velocity_warmup = False
    args.prob_t_eq_u = 0.0

    # Match steady-state post-warmup training behavior.
    args.lr_warmup_steps = 0
    args.update_ema_after = 0
    args.adaptive_weight_flowmap = bool(args.adaptive_weight_flowmap)
    args.use_edm_aug = bool(args.use_edm_aug)
    return args


def make_trainer(args: argparse.Namespace) -> Trainer:
    device = torch.device(args.device)
    model = instantiate_model(args).to(device)
    model.train(True)

    trainer = Trainer.__new__(Trainer)
    trainer.args = args
    trainer.device = device
    trainer.model = model
    trainer.model_without_ddp = model
    trainer.data_loader_train = (None,)
    trainer.model_restored = False
    trainer.update_steps = 0
    trainer.augment_pipe = AugmentPipe(
        p=0.12,
        xflip=1e8,
        yflip=0,
        scale=1,
        rotate_frac=0,
        aniso=1,
        translate_frac=1,
    )
    trainer.setup_optimizer()
    trainer.setup_compiled_forward_passes()
    return trainer


def benchmark_setup(
    benchmark_args: argparse.Namespace,
    method: str,
    stopgrad: str,
) -> dict[str, object]:
    torch.manual_seed(benchmark_args.seed)
    torch.cuda.manual_seed_all(benchmark_args.seed)
    args = training_args(benchmark_args, method, stopgrad)
    trainer = make_trainer(args)

    # Reuse a resident CUDA batch so the measurement reflects training compute,
    # not filesystem, DataLoader, or host-to-device variability.
    batch = torch.rand(
        args.batch_size,
        3,
        32,
        32,
        device=args.device,
    )

    for step in range(benchmark_args.warmup_steps):
        trainer.train_step(batch, data_idx=step, epoch_num=0)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(args.device)

    elapsed_seconds = []
    losses = []
    for step in range(benchmark_args.steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        stats = trainer.train_step(
            batch,
            data_idx=benchmark_args.warmup_steps + step,
            epoch_num=0,
        )
        torch.cuda.synchronize()
        elapsed_seconds.append(time.perf_counter() - start)
        losses.append(float(stats["loss"].cpu()))

    mean_seconds = statistics.mean(elapsed_seconds)
    row = {
        "method": method,
        "stopgrad": stopgrad,
        "batch_size": args.batch_size,
        "warmup_steps": benchmark_args.warmup_steps,
        "measured_steps": benchmark_args.steps,
        "mean_step_ms": mean_seconds * 1000,
        "median_step_ms": statistics.median(elapsed_seconds) * 1000,
        "std_step_ms": (
            statistics.pstdev(elapsed_seconds) * 1000
            if len(elapsed_seconds) > 1
            else 0.0
        ),
        "images_per_second": args.batch_size / mean_seconds,
        "peak_memory_gib": torch.cuda.max_memory_allocated(args.device)
        / (1024**3),
        "final_loss": losses[-1],
        "status": "ok",
    }

    del batch, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return row


def markdown_table(rows: list[dict[str, object]]) -> str:
    headers = (
        "method",
        "stopgrad",
        "mean step (ms)",
        "median (ms)",
        "std (ms)",
        "images/s",
        "peak GiB",
        "status",
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        if row["status"] == "ok":
            values = (
                row["method"],
                row["stopgrad"],
                f"{row['mean_step_ms']:.2f}",
                f"{row['median_step_ms']:.2f}",
                f"{row['std_step_ms']:.2f}",
                f"{row['images_per_second']:.2f}",
                f"{row['peak_memory_gib']:.2f}",
                row["status"],
            )
        else:
            values = (
                row["method"],
                row["stopgrad"],
                "",
                "",
                "",
                "",
                "",
                row["status"],
            )
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines)


def output_paths(requested_path: Path | None) -> tuple[Path, Path]:
    if requested_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        requested_path = (
            REPO_ROOT
            / "scripts"
            / "benchmark_results"
            / f"training_speed_{timestamp}.csv"
        )
    csv_path = requested_path.resolve()
    return csv_path, csv_path.with_suffix(".md")


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(
        f"batch={args.batch_size}, warmup={args.warmup_steps}, "
        f"measured={args.steps}, compile={args.compile}, "
        f"compile_second_order={args.compile_second_order}"
    )
    print("velocity warmup disabled; prob_t_eq_u=0.0")

    rows = []
    for method, stopgrad in SETUPS:
        label = f"{method}/{stopgrad}"
        print(f"\nBenchmarking {label}...", flush=True)
        try:
            row = benchmark_setup(args, method, stopgrad)
            print(
                f"{label}: {row['mean_step_ms']:.2f} ms/step, "
                f"{row['images_per_second']:.2f} images/s"
            )
        except Exception as error:
            row = {
                "method": method,
                "stopgrad": stopgrad,
                "batch_size": args.batch_size,
                "warmup_steps": args.warmup_steps,
                "measured_steps": args.steps,
                "mean_step_ms": "",
                "median_step_ms": "",
                "std_step_ms": "",
                "images_per_second": "",
                "peak_memory_gib": "",
                "final_loss": "",
                "status": f"error: {type(error).__name__}: {error}",
            }
            print(f"{label}: {row['status']}", file=sys.stderr)
            gc.collect()
            torch.cuda.empty_cache()
        rows.append(row)

    table = markdown_table(rows)
    print("\n" + table)

    csv_path, markdown_path = output_paths(args.output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    markdown_path.write_text(table + "\n")
    # Keep the configuration with the data rather than only in console output.
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    worktree = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(device),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_dirty": bool(worktree.stdout.strip()) if worktree.returncode == 0 else None,
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.steps,
        "compile": args.compile,
        "compile_second_order": args.compile_second_order,
        "seed": args.seed,
        "arch_name": "song_unet",
        "prob_t_eq_u": 0.0,
        "velocity_warmup_steps": 0,
        "memory_metric": "torch.cuda.max_memory_allocated (GiB), reset after warmup",
    }
    csv_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved table: {markdown_path}")
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
