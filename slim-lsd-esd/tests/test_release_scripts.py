"""Exercise the public launch/resume interface without GPUs or a Slurm allocation."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]


class ReleaseScriptsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.record = self.root / "arguments.json"
        self.python = self.root / "python-stub"
        self.python.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "if sys.argv[1].endswith('select_resume_checkpoint.py'):\n"
            "    print(os.environ.get('SELECTED_CHECKPOINT', ''))\n"
            "else:\n"
            "    with open(os.environ['ARG_RECORD'], 'w') as f:\n"
            "        json.dump(sys.argv[1:], f)\n"
        )
        self.python.chmod(0o755)
        self.env = {
            k: v for k, v in os.environ.items()
            if not k.startswith(('SLURM_', 'WANDB_', 'STOPGRAD_'))
            and k not in {'WHICH', 'SEED', 'STOPGRAD_MODE', 'COMPILE',
                          'COMPILE_SECOND_ORDER', 'VENV_PATH', 'USE_WANDB',
                          'RUN_NAME', 'RESUME_CHECKPOINT'}
        }
        self.env.update(
            PYTHON_BIN=str(self.python), ARG_RECORD=str(self.record),
            OUTPUT_ROOT=str(self.root / 'output with spaces'),
            DATA_ROOT=str(self.root / 'data with spaces'),
            RUN_NAME='test', GPUS='1', SEED='2', BATCH_SIZE='2',
        )

    def launch(self, **overrides):
        result = subprocess.run(
            ['bash', str(REPO_ROOT / 'scripts/train.sh'), '--max_train_steps', '2'],
            env={**self.env, **overrides}, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(self.record.read_text())

    def test_all_six_variants_compile_the_requested_objective(self):
        for method in ['eulerian', 'lagrangian']:
            for variant in ['none', 'orig', 'simp']:
                with self.subTest(method=method, variant=variant):
                    args = self.launch(WHICH=method, STOPGRAD_MODE=variant)
                    self.assertEqual(args[args.index('--which') + 1], method)
                    self.assertEqual(args[args.index('--stopgrad_type') + 1], variant)
                    self.assertEqual(args[args.index('--compile') + 1], '1')
                    self.assertEqual(args[args.index('--compile_second_order') + 1],
                                     '0' if variant == 'simp' else '1')
                    self.assertEqual(args[args.index('--output_dir') + 1], self.env['OUTPUT_ROOT'])
                    self.assertEqual(args[-2:], ['--max_train_steps', '2'])

    def test_mf_and_fm_do_not_use_esd_lsd_warmup(self):
        for method in ['meanflow', 'flowmatching']:
            with self.subTest(method=method):
                args = self.launch(WHICH=method)
                self.assertEqual(args[args.index('--velocity_warmup_steps') + 1], '0')

    def test_resume_uses_selected_checkpoint(self):
        checkpoint = str(self.root / 'checkpoint with spaces.pth')
        args = self.launch(SELECTED_CHECKPOINT=checkpoint)
        self.assertEqual(args[args.index('--resume') + 1], checkpoint)

    def test_eager_baseline_disables_second_order_compilation(self):
        args = self.launch(WHICH='eulerian', STOPGRAD_MODE='none', COMPILE='0')
        self.assertEqual(args[args.index('--compile_second_order') + 1], '0')

    def test_completed_run_is_skipped(self):
        ckpt = Path(self.env['OUTPUT_ROOT']) / 'test/0/2/ckpts'
        ckpt.mkdir(parents=True)
        (ckpt / 'training-complete').touch()
        result = subprocess.run(['bash', str(REPO_ROOT / 'scripts/train.sh')],
                                env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.record.exists())

    def test_requeue_without_checkpoint_fails(self):
        result = subprocess.run(
            ['bash', str(REPO_ROOT / 'scripts/train.sh')],
            env={**self.env, 'SLURM_JOB_ID': '123', 'SLURM_RESTART_COUNT': '1'},
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        self.assertFalse(self.record.exists())

    def test_slurm_template_hands_off_to_shared_launcher(self):
        srun = self.root / 'srun'
        srun.write_text('#!/bin/sh\nshift\nshift\nexec "$@"\n')
        srun.chmod(0o755)
        env = {**self.env, 'SLURM_JOB_ID': '123',
               'SLURM_SUBMIT_DIR': str(REPO_ROOT),
               'PATH': str(self.root) + os.pathsep + os.environ['PATH']}
        env.pop('REPO_ROOT', None)
        result = subprocess.run(
            ['bash', str(REPO_ROOT / 'scripts/train.sbatch')],
            env=env, cwd=self.root, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = json.loads(self.record.read_text())
        self.assertEqual(args[:2], ['-m', 'torch.distributed.run'])
        self.assertEqual(args[args.index('--job_id') + 1], '123')

    def test_selector_uses_newest_step_not_filename_order(self):
        ckpt = self.root / 'checkpoints'
        ckpt.mkdir()
        for step in [9, 100]:
            (ckpt / f'checkpoint-step{step}.pth').write_bytes(b'checkpoint')
        last = ckpt / 'checkpoint-last.pth'
        last.write_bytes(b'checkpoint')
        sidecar = ckpt / 'checkpoint-last.pth.step'
        sidecar.write_text('150\n')
        last_time = last.stat().st_mtime_ns
        os.utime(sidecar, ns=(last_time + 1, last_time + 1))
        result = subprocess.run(
            ['python', str(REPO_ROOT / 'scripts/select_resume_checkpoint.py'), str(ckpt)],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(last))


if __name__ == '__main__':
    unittest.main()
