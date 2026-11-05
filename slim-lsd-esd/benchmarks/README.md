# Compiled training benchmark

[compiled_b200.csv](compiled_b200.csv) contains measurements for all six
ESD/LSD objectives with compilation enabled. The `none` and `orig` variants
use the second-order compilation compatibility setting.

| Objective | No stopgrad | Original stopgrad | Slim |
| --- | ---: | ---: | ---: |
| ESD peak GiB | 64.603 | 50.005 | 31.226 |
| LSD peak GiB | 60.753 | 40.998 | 26.845 |
| ESD mean step ms | 304.064 | 311.932 | 253.270 |
| LSD mean step ms | 296.691 | 257.775 | 200.596 |

Slim uses 2.07x less memory for ESD and 2.26x less for LSD than the objectives
without stopgrad, and about 1.5–1.6x less than the original stopgrad objectives.

The run used one NVIDIA B200, batch size 128, synthetic 3x32x32 inputs and a
Song U-Net. Two compilation/warmup steps preceded five measured optimizer
steps. Velocity warmup was off and `prob_t_eq_u=0.0`. Timings include forward,
backward, optimizer and EMA updates; data loading, evaluation, checkpointing
and W&B are excluded. Memory is peak allocated CUDA memory in GiB, reset
after warmup. Timing standard deviations in the CSV are over the five steps.

The CSV does not record the Git commit or software versions. The
[benchmark script](../scripts/benchmark_training_speed.py) writes those to a
JSON sidecar. See the [training README](../README.md#benchmarking) for the command.
