# StopGrad

Code for *How I learned to stop worrying and love StopGrads: Stationarity,
Convergence, and a case study on Flow Map Learning*.

- [Numerical experiments](numerical_experiments/README.md): simulate Eulerian
  and Lagrangian functional gradient flow on a discrete grid and compare with
  the explicit solutions. Includes 1D and 2D reference figures.
- [Slim ESD and LSD](slim-lsd-esd/README.md): train and evaluate slim ESD and LSD flow map models on CIFAR-10.

Each directory has its own dependencies and run instructions. The numerical
experiments run on CPU or CUDA.

## Citation

[NeurIPS 2026](https://neurips.cc/virtual/2026/poster/149145) | [arXiv](https://arxiv.org/abs/2609.16222)

```bibtex
@inproceedings{shen2026stopgrads,
  title = {How I learned to stop worrying and love {StopGrads}: Stationarity, Convergence, and a case study on Flow Map Learning},
  author = {Max W. Shen and Mark Goldstein and Zichu Wang and Aahlad Puli and Rajesh Ranganath},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2026},
  url = {https://neurips.cc/virtual/2026/poster/149145}
}
```
