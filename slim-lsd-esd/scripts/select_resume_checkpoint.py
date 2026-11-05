#!/usr/bin/env python3
"""Print the newest committed checkpoint in a stopgrad checkpoint directory."""

import re
import sys
from pathlib import Path


PERIODIC_RE = re.compile(r"checkpoint-step([0-9]+)\.pth$")


def read_last_step(checkpoint_path: Path) -> int | None:
    sidecar = checkpoint_path.with_suffix(checkpoint_path.suffix + ".step")
    try:
        # A sidecar older than the checkpoint can be left behind if a process
        # dies between replacing the checkpoint and replacing its metadata.
        if sidecar.stat().st_mtime_ns >= checkpoint_path.stat().st_mtime_ns:
            return int(sidecar.read_text().strip())
    except (FileNotFoundError, ValueError):
        pass

    # Older checkpoints predate the lightweight step sidecar. Inspect one such
    # checkpoint once; subsequent saves create the sidecar.
    try:
        import torch

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        return int(checkpoint["update_steps"])
    except Exception as error:
        print(
            f"Warning: ignoring unreadable checkpoint {checkpoint_path}: {error}",
            file=sys.stderr,
        )
        return None


def select_checkpoint(checkpoint_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []

    for path in checkpoint_dir.glob("checkpoint-step*.pth"):
        match = PERIODIC_RE.fullmatch(path.name)
        if match and path.is_file() and path.stat().st_size > 0:
            candidates.append((int(match.group(1)), path))

    last_path = checkpoint_dir / "checkpoint-last.pth"
    if last_path.is_file() and last_path.stat().st_size > 0:
        last_step = read_last_step(last_path)
        if last_step is not None:
            candidates.append((last_step, last_path))

    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate[0])[1]


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} CKPT_DIR", file=sys.stderr)
        return 2
    checkpoint = select_checkpoint(Path(sys.argv[1]))
    if checkpoint is not None:
        print(checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
