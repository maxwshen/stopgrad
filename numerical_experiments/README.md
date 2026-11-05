# Numerical convergence experiments

We simulate Eulerian and Lagrangian functional gradient flow on a discrete
grid and compare the result with the explicit solutions. The
[methods note](sim_func_grad_flow/sim_functional_grad_flow.md) describes the
grid, velocity field, solver and error measurements.

## Running the experiments

From the repository root, install the dependencies in a Python 3.11 or later
environment:

```bash
python -m pip install -r numerical_experiments/requirements.txt
```

Run the 1D experiment with seven pointwise curves through optimization time 5:

```bash
python numerical_experiments/sim_func_grad_flow/sim_functional_grad_flow.py \
  --mode both --device cpu --plot_kind both \
  --num_f0s 1 --pointwise_num_curves 7 --pointwise_tau_final 5 \
  --output_dir outputs/convergence
```

Add `--spatial_dim 2` for 2D, or use `--device cuda` for a GPU. `--help` lists
the grid, time-step, initial-condition and plotting options. For a quick check:

```bash
python numerical_experiments/sim_func_grad_flow/sim_functional_grad_flow.py \
  --mode both --device cpu --n_tu 9 --n_x 13 \
  --tau_final 0.1 --dtau 0.005 --num_f0s 2 --checkpoints 4 \
  --plot_kind both --pointwise_num_curves 2 --output_dir outputs/convergence-smoke
```

Each mode writes a `summary.json` file and the plots selected by `--plot_kind`:
checkpoint errors against the explicit solution, a solution scatter plot, and
linear/log pointwise error curves. Outputs are grouped by dimension and mode.
Running both modes also writes a combined scatter plot.

## Reference figures

[reference_figures/](reference_figures/README.md) contains 1D and 2D outputs,
their summaries and the commands used to generate them.

The code has been checked with Python 3.11.15, PyTorch 2.10.0+cu128,
NumPy 2.4.2, Matplotlib 3.10.8 and tqdm 4.67.3. Checks covered both modes on
CPU and CUDA, plot generation, and reduced error under grid refinement.
