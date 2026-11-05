"""Discrete-grid simulation of Eulerian and Lagrangian functional gradient flow.

Finite differences and RK4 integration use PyTorch tensors on CPU or CUDA.
Exact affine flow maps provide reference solutions for convergence diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from tqdm import tqdm

Mode = Literal["eulerian", "lagrangian"]

PLOT_FONT_SIZES = {
    # "font.size": 19.5,
    # "axes.titlesize": 22.5,
    # "axes.labelsize": 21,
    # "xtick.labelsize": 18,
    # "ytick.labelsize": 18,
    # "legend.fontsize": 18,
    # "figure.titlesize": 24,
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 16,
}

plt.rcParams.update(PLOT_FONT_SIZES)


@dataclass
class AffineVelocityTorch:
    """Constant-coefficient affine velocity field v(x) = A x + b."""

    A: torch.Tensor
    b: torch.Tensor
    _flow_cache: Dict[Tuple[float, str, str], Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.A = torch.as_tensor(self.A)
        self.b = torch.as_tensor(self.b)
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A must be square.")
        if self.b.shape != (self.A.shape[0],):
            raise ValueError("b must have shape (dim,).")
        self.dim = int(self.b.shape[0])

    def v(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ij,...j->...i", self.A.to(device=x.device, dtype=x.dtype), x) + self.b.to(device=x.device, dtype=x.dtype)

    def lipschitz_constant(self) -> float:
        return float(torch.linalg.matrix_norm(self.A, ord=2).item())

    def _flow_matrix(self, dt: float, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (round(float(dt), 12), str(device), str(dtype))
        if key not in self._flow_cache:
            dim = self.dim
            B = torch.zeros((dim + 1, dim + 1), device=device, dtype=dtype)
            B[:dim, :dim] = self.A.to(device=device, dtype=dtype)
            B[:dim, -1] = self.b.to(device=device, dtype=dtype)
            M_aug = torch.matrix_exp(float(dt) * B)
            M = M_aug[:dim, :dim]
            c = M_aug[:dim, -1]
            self._flow_cache[key] = (M, c)
        return self._flow_cache[key]

    def flow(self, t: torch.Tensor, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        dt = (u - t).expand(x.shape[:-1])
        flat_dt = torch.round(dt.reshape(-1) * 1e12) / 1e12
        flat_x = x.reshape(-1, self.dim)
        flat_out = torch.empty_like(flat_x)

        unique_dt = torch.unique(flat_dt)
        for val in unique_dt:
            mask = flat_dt == val
            M, c = self._flow_matrix(float(val.item()), device=x.device, dtype=x.dtype)
            flat_out[mask] = flat_x[mask] @ M.T + c

        return flat_out.reshape(x.shape)


def make_grid(
    dim: int,
    n_tu: int,
    n_x: int,
    xlim: Tuple[float, float],
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor | Tuple[torch.Tensor, ...] | Tuple[int, ...] | int]:
    if dim < 1 or dim > 5:
        raise ValueError("Only spatial dimensions 1 through 5 are supported.")

    t = torch.linspace(0.0, 1.0, n_tu, device=device, dtype=dtype)
    u = torch.linspace(0.0, 1.0, n_tu, device=device, dtype=dtype)
    x_axes = tuple(torch.linspace(xlim[0], xlim[1], n_x, device=device, dtype=dtype) for _ in range(dim))
    spatial_shape = tuple(n_x for _ in range(dim))
    mesh = torch.meshgrid(*x_axes, indexing="ij")
    X = torch.stack(mesh, dim=-1).unsqueeze(0).unsqueeze(0)
    q = (u[None, :] - t[:, None]).reshape((len(t), len(u)) + (1,) * (dim + 1))
    mask = u[None, :] >= t[:, None]
    diag_index = torch.arange(len(t), device=device)

    return {
        "dim": dim,
        "t": t,
        "u": u,
        "q": q,
        "X": X,
        "x_axes": x_axes,
        "spatial_shape": spatial_shape,
        "mask": mask,
        "diag_index": diag_index,
    }


def broadcast_coordinates(grid: Dict[str, object]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = grid["t"]
    u = grid["u"]
    dim = int(grid["dim"])
    spatial_shape = tuple(grid["spatial_shape"])
    expand_shape = (len(t), len(u)) + spatial_shape + (dim,)
    T = t.reshape((len(t),) + (1,) * (dim + 1))
    U = u.reshape((1, len(u)) + (1,) * dim)
    X = torch.broadcast_to(grid["X"], expand_shape)
    return T, U, X


def analytic_initial_map(
    t: torch.Tensor,
    u: torch.Tensor,
    x: torch.Tensor,
    velocity: AffineVelocityTorch,
    eps: float,
    f0_index: int,
) -> torch.Tensor:
    base = velocity.flow(t, u, x)
    lead_shape = x.shape[:-1]
    t_b = torch.broadcast_to(t, lead_shape)
    u_b = torch.broadcast_to(u, lead_shape)
    q = u_b - t_b
    phase_t = 0.35 * f0_index
    phase_u = 0.27 * f0_index
    phase_x = 0.19 * f0_index
    amp = eps * (1.0 + 0.08 * f0_index)

    deltas = []
    for coord_idx in range(velocity.dim):
        x_i = x[..., coord_idx]
        x_next = x[..., (coord_idx + 1) % velocity.dim]
        trig_time = torch.sin((coord_idx + 1) * torch.pi * t_b + phase_t) * torch.cos((coord_idx + 1) * torch.pi * u_b + phase_u)
        trig_space = torch.sin((0.80 + 0.10 * coord_idx) * x_i + phase_x)
        cross_space = torch.cos((0.60 + 0.05 * coord_idx) * x_next - phase_x)
        mixed_time = torch.cos((coord_idx + 1) * torch.pi * (t_b + u_b) + phase_t - phase_u)
        local_space = torch.cos((0.70 + 0.07 * coord_idx) * x_i - phase_x)
        delta_i = amp * q * (0.60 * trig_time * trig_space * cross_space + 0.35 * mixed_time * local_space)
        deltas.append(delta_i)
    return base + torch.stack(deltas, dim=-1)


def exact_rectified_solution(
    mode: Mode,
    tau: float,
    grid: Dict[str, object],
    velocity: AffineVelocityTorch,
    eps: float,
    f0_index: int,
) -> torch.Tensor:
    T, U, X = broadcast_coordinates(grid)
    if mode == "eulerian":
        s = U - torch.exp(torch.tensor(-tau, device=X.device, dtype=X.dtype)) * (U - T)
        x_in = velocity.flow(T, s, X)
        return analytic_initial_map(s, U, x_in, velocity, eps=eps, f0_index=f0_index)
    if mode == "lagrangian":
        s = T + torch.exp(torch.tensor(-tau, device=X.device, dtype=X.dtype)) * (U - T)
        y = analytic_initial_map(T, s, X, velocity, eps=eps, f0_index=f0_index)
        return velocity.flow(s, U, y)
    raise ValueError(f"Unknown mode: {mode}")


def exact_rectified_point(
    mode: Mode,
    tau: float,
    t: torch.Tensor,
    u: torch.Tensor,
    x: torch.Tensor,
    velocity: AffineVelocityTorch,
    eps: float,
    f0_index: int,
) -> torch.Tensor:
    """Evaluate the explicit rectified formula at one fixed probe point."""
    tau_t = torch.tensor(-tau, device=x.device, dtype=x.dtype)
    if mode == "eulerian":
        s = u - torch.exp(tau_t) * (u - t)
        x_in = velocity.flow(t, s, x)
        return analytic_initial_map(s, u, x_in, velocity, eps=eps, f0_index=f0_index)
    if mode == "lagrangian":
        s = t + torch.exp(tau_t) * (u - t)
        y = analytic_initial_map(t, s, x, velocity, eps=eps, f0_index=f0_index)
        return velocity.flow(s, u, y)
    raise ValueError(f"Unknown mode: {mode}")


def eulerian_rhs(
    f: torch.Tensor,
    grid: Dict[str, object],
    velocity: AffineVelocityTorch,
) -> torch.Tensor:
    t = grid["t"]
    x_axes = grid["x_axes"]
    dt = float((t[1] - t[0]).item())
    dt_f = torch.gradient(f, spacing=(dt,), dim=(0,), edge_order=2)[0]
    X = torch.broadcast_to(grid["X"], f.shape)
    v_on_grid = velocity.v(X)
    jacobian_vec = torch.zeros_like(f)
    for coord_idx, axis_values in enumerate(x_axes):
        dx = float((axis_values[1] - axis_values[0]).item())
        dfdx_i = torch.gradient(f, spacing=(dx,), dim=(2 + coord_idx,), edge_order=2)[0]
        jacobian_vec = jacobian_vec + dfdx_i * v_on_grid[..., coord_idx][..., None]
    return grid["q"] * (dt_f + jacobian_vec)


def lagrangian_rhs(
    f: torch.Tensor,
    grid: Dict[str, object],
    velocity: AffineVelocityTorch,
) -> torch.Tensor:
    u = grid["u"]
    du = float((u[1] - u[0]).item())
    du_f = torch.gradient(f, spacing=(du,), dim=(1,), edge_order=2)[0]
    vf = velocity.v(f)
    return -grid["q"] * (du_f - vf)


def enforce_diagonal_identity(f: torch.Tensor, grid: Dict[str, object]) -> None:
    X_diag = grid["X"][0, 0]
    diag_index = grid["diag_index"]
    f[diag_index, diag_index] = X_diag


def upper_triangle_errors(f_num: torch.Tensor, f_ref: torch.Tensor, grid: Dict[str, object]) -> Tuple[float, float]:
    diff = f_num - f_ref
    point_sq = torch.sum(diff * diff, dim=-1)
    extra = (None,) * len(grid["spatial_shape"])
    mask = torch.broadcast_to(grid["mask"][(slice(None), slice(None)) + extra], point_sq.shape)
    l2 = torch.sqrt(torch.mean(point_sq[mask]))
    linf = torch.sqrt(torch.max(point_sq[mask]))
    return float(l2.item()), float(linf.item())


def rk4_integrate(
    mode: Mode,
    grid: Dict[str, object],
    velocity: AffineVelocityTorch,
    tau_final: float,
    dtau: float,
    eps: float,
    checkpoints: int,
    f0_index: int,
    compute_history: bool = True,
    store_snapshots: bool = True,
) -> Tuple[torch.Tensor, np.ndarray, List[Dict[str, object]]]:
    T, U, X = broadcast_coordinates(grid)
    f = analytic_initial_map(T, U, X, velocity, eps=eps, f0_index=f0_index).clone()
    enforce_diagonal_identity(f, grid)
    rhs = eulerian_rhs if mode == "eulerian" else lagrangian_rhs

    nsteps = int(round(tau_final / dtau))
    checkpoint_every = max(1, nsteps // max(1, checkpoints))
    tau = 0.0
    history = []
    snapshots: List[Dict[str, object]] = []

    for step in tqdm(range(nsteps + 1), desc='rk4 integration'):
        should_checkpoint = step % checkpoint_every == 0 or step == nsteps
        if should_checkpoint and compute_history:
            f_exact = exact_rectified_solution(mode, tau, grid, velocity, eps=eps, f0_index=f0_index)
            l2, linf = upper_triangle_errors(f, f_exact, grid)
            history.append((tau, l2, linf))
            if store_snapshots:
                snapshots.append(
                    {
                        "tau": float(tau),
                        "f0_index": int(f0_index),
                        # Offload snapshots immediately to CPU; later diagnostics
                        # are CPU-side and there is no reason to retain every
                        # checkpoint tensor on GPU.
                        "f_num": f.detach().cpu().clone(),
                        "f_exact": f_exact.detach().cpu().clone(),
                    }
                )
            del f_exact
        if step == nsteps:
            break

        k1 = rhs(f, grid, velocity)
        k2 = rhs(f + 0.5 * dtau * k1, grid, velocity)
        k3 = rhs(f + 0.5 * dtau * k2, grid, velocity)
        k4 = rhs(f + dtau * k3, grid, velocity)
        f = f + (dtau / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        del k1, k2, k3, k4
        enforce_diagonal_identity(f, grid)
        tau += dtau

    return f, np.asarray(history, dtype=float), snapshots


def recommend_dtau(grid: Dict[str, object], velocity: AffineVelocityTorch, cfl: float = 0.20) -> float:
    spacings = [float((grid["t"][1] - grid["t"][0]).item()), float((grid["u"][1] - grid["u"][0]).item())]
    spacings.extend(float((axis[1] - axis[0]).item()) for axis in grid["x_axes"])
    X = torch.broadcast_to(grid["X"], (1, 1, *grid["spatial_shape"], velocity.dim))
    vmax = float(torch.max(torch.linalg.vector_norm(velocity.v(X), dim=-1)).item())
    denom = 1.0 + vmax + velocity.lipschitz_constant()
    return cfl * min(spacings) / denom


def plot_error_history(history: np.ndarray, title: str, output_path: str) -> None:
    tau = history[:, 0]
    l2 = history[:, 1]
    linf = history[:, 2]
    plt.figure(figsize=(6.0, 4.2))
    plt.semilogy(tau, l2, marker="o", label=r"$L^2$-type error")
    plt.semilogy(tau, linf, marker="s", label=r"$L^\infty$-type error")
    plt.xlabel(r"optimization time $\tau$")
    plt.ylabel("error vs. exact rectified formula")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 0.0 or y_std <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def build_solution_scatter_data(
    snapshots: List[Dict[str, object]],
    grid: Dict[str, object],
    mode: Mode,
    num_f0s: int,
    max_points: int = 40000,
) -> Dict[str, object]:
    extra = (None,) * len(grid["spatial_shape"])
    point_mask = torch.broadcast_to(
        grid["mask"][(slice(None), slice(None)) + extra],
        snapshots[0]["f_num"].shape[:-1],
    ).detach().cpu()

    exact_values = []
    num_values = []
    use_norms = int(grid["dim"]) > 1
    for snapshot in snapshots:
        # Avoid CUDA advanced-indexing asserts on large broadcasted boolean masks.
        f_num = snapshot["f_num"].detach().cpu()
        f_exact = snapshot["f_exact"].detach().cpu()
        exact_selected = f_exact[point_mask]
        num_selected = f_num[point_mask]
        if use_norms:
            exact_values.append(torch.linalg.vector_norm(exact_selected, dim=-1).reshape(-1))
            num_values.append(torch.linalg.vector_norm(num_selected, dim=-1).reshape(-1))
        else:
            exact_values.append(exact_selected.reshape(-1))
            num_values.append(num_selected.reshape(-1))

    x_t = torch.cat(exact_values)
    y_t = torch.cat(num_values)
    total_points = int(x_t.numel())
    if x_t.numel() > max_points:
        idx = torch.arange(max_points, dtype=torch.float64)
        idx = torch.floor(idx * (x_t.numel() / max_points)).long()
        x_t = x_t[idx]
        y_t = y_t[idx]

    x = x_t.detach().cpu().numpy()
    y = y_t.detach().cpu().numpy()
    if mode == "eulerian":
        base_xlabel = r"$F_0\!\left(s_{t\to u}^{\tau}, u, F^*(t, s_{t\to u}^{\tau}, x)\right)$"
    else:
        base_xlabel = r"$F^*\!\left(s_{u\to t}^{\tau}, u, F_0(t, s_{u\to t}^{\tau}, x)\right)$"
    xlabel = rf"$\|{base_xlabel[1:-1]}\|_2$" if use_norms else base_xlabel
    ylabel = r"$\|f_\tau(t,u,x)\|_2$" if use_norms else r"$f_\tau(t,u,x)$"

    return {
        "x": x,
        "y": y,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "pearsonr": pearsonr(x, y),
        "num_f0_values": int(num_f0s),
        "num_tau_values": int(len(snapshots)),
        "num_t_values": int(len(grid["t"])),
        "num_x_points": int(np.prod(grid["spatial_shape"])),
        "total_points": total_points,
        "uses_norms": use_norms,
    }


def build_pointwise_probe_specs(
    grid: Dict[str, object],
    num_f0s: int,
    num_curves: int,
) -> List[Dict[str, object]]:
    """Choose deterministic probe points `(f0, t, u, x)` on the discrete grid."""
    if num_curves <= 0:
        return []

    t = grid["t"]
    u = grid["u"]
    spatial_shape = tuple(int(v) for v in grid["spatial_shape"])
    valid_pairs = [(ti, ui) for ti in range(len(t)) for ui in range(ti + 1, len(u))]
    if not valid_pairs:
        valid_pairs = [(0, 0)]

    n_x = spatial_shape[0]
    candidate_positions = []
    for raw in (n_x // 6, n_x // 4, n_x // 2, (3 * n_x) // 4, (5 * n_x) // 6):
        idx = min(max(raw, 0), n_x - 1)
        if idx not in candidate_positions:
            candidate_positions.append(idx)
    if not candidate_positions:
        candidate_positions = [0]

    specs: List[Dict[str, object]] = []
    for curve_idx in range(num_curves):
        pair_idx = int(round(curve_idx * (len(valid_pairs) - 1) / max(num_curves - 1, 1)))
        t_idx, u_idx = valid_pairs[pair_idx]
        f0_index = curve_idx % max(num_f0s, 1)
        x_index = tuple(
            candidate_positions[(curve_idx + coord_idx) % len(candidate_positions)]
            for coord_idx in range(len(spatial_shape))
        )
        x_coords = [float(grid["x_axes"][coord_idx][x_index[coord_idx]].item()) for coord_idx in range(len(spatial_shape))]
        specs.append(
            {
                "curve_index": curve_idx,
                "f0_index": int(f0_index),
                "t_index": int(t_idx),
                "u_index": int(u_idx),
                "x_index": x_index,
                "t_value": float(t[t_idx].item()),
                "u_value": float(u[u_idx].item()),
                "x_value": x_coords,
            }
        )
    return specs


def integrate_pointwise_error_curves(
    mode: Mode,
    grid: Dict[str, object],
    velocity: AffineVelocityTorch,
    tau_final: float,
    dtau: float,
    eps: float,
    num_f0s: int,
    num_curves: int,
) -> List[Dict[str, object]]:
    """Track fixed-point numerical and exact pointwise errors along the tau flow."""
    probe_specs = build_pointwise_probe_specs(grid=grid, num_f0s=num_f0s, num_curves=num_curves)
    if not probe_specs:
        return []

    curves_by_f0: Dict[int, List[Dict[str, object]]] = {}
    for spec in probe_specs:
        curve = {
            **spec,
            "tau_values": [],
            "numerical_error": [],
            "predicted_error": [],
        }
        curves_by_f0.setdefault(int(spec["f0_index"]), []).append(curve)

    rhs = eulerian_rhs if mode == "eulerian" else lagrangian_rhs
    T, U, X = broadcast_coordinates(grid)
    nsteps = int(round(tau_final / dtau))

    for f0_index in range(num_f0s):
        f = analytic_initial_map(T, U, X, velocity, eps=eps, f0_index=f0_index).clone()
        enforce_diagonal_identity(f, grid)
        tau = 0.0
        curves = curves_by_f0.get(f0_index, [])
        progress = tqdm(
            range(nsteps + 1),
            desc=f"Pointwise error ({mode}, f0={f0_index})",
            dynamic_ncols=True,
        )
        for step in progress:
            if curves:
                for curve in curves:
                    x_key = tuple(int(v) for v in curve["x_index"])
                    num_point = f[(int(curve["t_index"]), int(curve["u_index"])) + x_key + (slice(None),)]
                    t_val = grid["t"][int(curve["t_index"])]
                    u_val = grid["u"][int(curve["u_index"])]
                    x_coords = torch.stack(
                        [grid["x_axes"][coord_idx][x_key[coord_idx]] for coord_idx in range(len(x_key))]
                    )
                    target = velocity.flow(t_val, u_val, x_coords)
                    predicted = exact_rectified_point(
                        mode=mode,
                        tau=tau,
                        t=t_val,
                        u=u_val,
                        x=x_coords,
                        velocity=velocity,
                        eps=eps,
                        f0_index=f0_index,
                    )
                    curve["tau_values"].append(float(tau))
                    curve["numerical_error"].append(float(torch.linalg.vector_norm(num_point - target).item()))
                    curve["predicted_error"].append(float(torch.linalg.vector_norm(predicted - target).item()))

            if step == nsteps:
                break

            k1 = rhs(f, grid, velocity)
            k2 = rhs(f + 0.5 * dtau * k1, grid, velocity)
            k3 = rhs(f + 0.5 * dtau * k2, grid, velocity)
            k4 = rhs(f + dtau * k3, grid, velocity)
            f = f + (dtau / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            enforce_diagonal_identity(f, grid)
            tau += dtau

    result = []
    for f0_curves in curves_by_f0.values():
        for curve in f0_curves:
            result.append(curve)
    return sorted(result, key=lambda curve: int(curve["curve_index"]))


def plot_pointwise_error_curves(
    curves: List[Dict[str, object]],
    mode: Mode,
    title: str,
    output_path: str,
    logy: bool,
) -> None:
    """Plot numerical and exact pointwise error curves against tau."""
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
    if mode == "eulerian":
        predicted_label = r"$\|F_0(s_{t\to u}^{\tau}, u, F^*(t, s_{t\to u}^{\tau}, x)) - F^*(t,u,x)\|_2$"
    else:
        predicted_label = r"$\|F^*(s_{u\to t}^{\tau}, u, F_0(t, s_{u\to t}^{\tau}, x)) - F^*(t,u,x)\|_2$"
    style_handles = [
        Line2D([0], [0], color="black", linewidth=2.0, alpha=0.5, 
               label=r"$\|F_\tau(t,u,x)-F^*(t,u,x)\|_2$"
        ),
        Line2D([0], [0], color="black", linewidth=1.8, linestyle="--", label=predicted_label),
        Line2D([0], [0], color="C0", linewidth=2.0, label=r"Colors indicate distinct $(F_0, t, u, x)$"),
    ]
    for curve_idx, curve in enumerate(curves):
        color = colors[curve_idx % len(colors)]
        tau_values = np.asarray(curve["tau_values"], dtype=float)
        numerical_error = np.asarray(curve["numerical_error"], dtype=float)
        predicted_error = np.asarray(curve["predicted_error"], dtype=float)
        ax.plot(tau_values, numerical_error, color=color, linewidth=2.0, alpha=0.5)
        ax.plot(tau_values, predicted_error, color=color, linestyle="--", linewidth=1.8)

    ax.set_xlabel(r"Optimization time $\tau$")
    ax.set_ylabel(r"Flow map error")
    # ax.set_ylabel(r"$\|f_\tau(t,u,x) - f^*(t,u,x)\|_2$")
    # ax.set_ylabel(r"$\|f_\tau(t,u,x) - f^*(t,u,x)\|_2$", rotation=0, ha='right')
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(handles=style_handles, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return


def plot_solution_scatter(scatter_data: Dict[str, object], title: str, output_path: str) -> float:
    x = np.asarray(scatter_data["x"])
    y = np.asarray(scatter_data["y"])
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    pad = 0.05 * max(hi - lo, 1e-12)
    lo -= pad
    hi += pad

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.scatter(x, y, s=8, alpha=0.35)
    ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=2, alpha=0.35, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(str(scatter_data["xlabel"]))
    ax.set_ylabel(str(scatter_data["ylabel"]), ha='right')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()

    r = float(scatter_data["pearsonr"])
    stats_label = "\n".join(
        [
            f"pearsonr = {r:.6f}" if np.isfinite(r) else "pearsonr = nan",
            f"num. f0 values = {scatter_data['num_f0_values']}",
            f"num. tau values = {scatter_data['num_tau_values']}",
            f"num. t/u values = {scatter_data['num_t_values']}",
            f"num. x points = {scatter_data['num_x_points']}",
            f"total points = {scatter_data['total_points']}",
        ]
    )
    ax.text(
        0.98,
        0.04,
        stats_label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2.5},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return r


def plot_combined_solution_scatter(scatter_by_mode: Dict[str, Dict[str, object]], title: str, output_path: str) -> None:
    modes = list(scatter_by_mode.keys())
    fig, axes = plt.subplots(1, len(modes), figsize=(6.4 * len(modes), 5.0), squeeze=False)

    all_vals = []
    for mode in modes:
        all_vals.append(np.asarray(scatter_by_mode[mode]["x"]))
        all_vals.append(np.asarray(scatter_by_mode[mode]["y"]))
    merged = np.concatenate(all_vals)
    lo = float(np.min(merged))
    hi = float(np.max(merged))
    pad = 0.05 * max(hi - lo, 1e-12)
    lo -= pad
    hi += pad

    for idx, mode in enumerate(modes):
        ax = axes[0, idx]
        scatter_data = scatter_by_mode[mode]
        x = np.asarray(scatter_data["x"])
        y = np.asarray(scatter_data["y"])
        ax.scatter(x, y, s=8, alpha=0.35)
        ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=2, alpha=0.35, label="y = x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(str(scatter_data["xlabel"]))
        if idx == 0:
            ax.set_ylabel(str(scatter_data["ylabel"]), ha='right')
        ax.set_title(mode.title())
        ax.grid(alpha=0.3)
        ax.legend()
        r = float(scatter_data["pearsonr"])
        stats_label = "\n".join(
            [
                f"pearsonr = {r:.6f}" if np.isfinite(r) else "pearsonr = nan",
                f"num. f0 values = {scatter_data['num_f0_values']}",
                f"num. tau values = {scatter_data['num_tau_values']}",
                f"num. t/u values = {scatter_data['num_t_values']}",
                f"num. x points = {scatter_data['num_x_points']}",
                f"total points = {scatter_data['total_points']}",
            ]
        )
        ax.text(
            0.98,
            0.04,
            stats_label,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2.5},
        )

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def default_velocity(dim: int, device: torch.device, dtype: torch.dtype) -> AffineVelocityTorch:
    if dim < 1 or dim > 5:
        raise ValueError("Only spatial dimensions 1 through 5 are supported.")
    A = torch.zeros((dim, dim), device=device, dtype=dtype)
    for i in range(dim):
        A[i, i] = 0.15 + 0.05 / (i + 1)
        if i + 1 < dim:
            A[i, i + 1] = -0.04 / (i + 1)
            A[i + 1, i] = 0.03 / (i + 1)
    b = torch.linspace(0.10, -0.02, dim, device=device, dtype=dtype)
    return AffineVelocityTorch(A=A, b=b)


def run_demo(
    dim: int,
    mode: Mode,
    device: torch.device,
    dtype: torch.dtype,
    n_tu: int | None = None,
    n_x: int | None = None,
    xlim: Tuple[float, float] = (-2.0, 2.0),
    tau_final: float = 0.20,
    dtau: float | None = None,
    eps: float = 0.15,
    num_f0s: int = 1,
    checkpoints: int = 20,
    plot_kind: str = "both",
    pointwise_tau_final: float | None = None,
    pointwise_dtau: float | None = None,
    pointwise_num_curves: int = 1,
    output_dir: str | None = None,
) -> Tuple[
    torch.Tensor,
    np.ndarray,
    Dict[str, object],
    AffineVelocityTorch,
    Dict[str, float | int | object],
    Dict[str, object] | None,
]:
    if n_tu is None:
        n_tu = 61 if dim == 1 else 31 if dim == 2 else 21
    if n_x is None:
        n_x = 121 if dim == 1 else 31 if dim == 2 else 9

    velocity = default_velocity(dim, device=device, dtype=dtype)
    grid = make_grid(dim=dim, n_tu=n_tu, n_x=n_x, xlim=xlim, device=device, dtype=dtype)

    if dtau is None:
        dtau = recommend_dtau(grid, velocity, cfl=0.20)
    if pointwise_tau_final is None:
        pointwise_tau_final = tau_final
    if pointwise_dtau is None:
        pointwise_dtau = dtau

    nsteps = int(round(tau_final / dtau))
    history_list = []
    snapshots: List[Dict[str, object]] = []
    f_final = None
    need_scatter = plot_kind in {"scatter", "both"}
    need_history = plot_kind in {"scatter", "both"}
    if need_scatter or need_history:
        for f0_index in range(num_f0s):
            f_final_i, history_i, snapshots_i = rk4_integrate(
                mode=mode,
                grid=grid,
                velocity=velocity,
                tau_final=tau_final,
                dtau=dtau,
                eps=eps,
                checkpoints=checkpoints,
                f0_index=f0_index,
                compute_history=need_history,
                store_snapshots=need_scatter,
            )
            if need_history and history_i.size > 0:
                history_list.append(history_i)
            if need_scatter:
                snapshots.extend(snapshots_i)
            f_final = f_final_i
        if need_history and history_list:
            history = np.mean(np.stack(history_list, axis=0), axis=0)
        else:
            history = np.asarray([[0.0, np.nan, np.nan]], dtype=float)
    else:
        history = np.asarray([[0.0, np.nan, np.nan]], dtype=float)

    scatter_data = None
    scatter_pearsonr = float("nan")
    pointwise_curves: List[Dict[str, object]] = []
    if output_dir is not None:
        if need_history:
            plot_error_history(history, f"{mode.title()}  |  dim={dim}  |  dtau={dtau:.2e}", os.path.join(output_dir, "error_history.png"))
        if need_scatter:
            scatter_data = build_solution_scatter_data(snapshots=snapshots, grid=grid, mode=mode, num_f0s=num_f0s)
            scatter_pearsonr = plot_solution_scatter(
                scatter_data=scatter_data,
                title=f"{mode.title()}",
                output_path=os.path.join(output_dir, f"solution_scatter_{dim}d_{mode.title()}_{num_f0s}nf0s.png"),
            )
        if plot_kind in {"pointwise_error", "both"} and pointwise_num_curves > 0:
            pointwise_curves = integrate_pointwise_error_curves(
                mode=mode,
                grid=grid,
                velocity=velocity,
                tau_final=pointwise_tau_final,
                dtau=pointwise_dtau,
                eps=eps,
                num_f0s=num_f0s,
                num_curves=pointwise_num_curves,
            )
            plot_pointwise_error_curves(
                curves=pointwise_curves,
                mode=mode,
                title=f"{mode.title()}",
                output_path=os.path.join(output_dir, f"pointwise_log_error_{dim}d_{mode.title()}_tau{pointwise_tau_final}_curves{pointwise_num_curves}_ntu{n_tu}_nx{n_x}.png"),
                logy=True,
            )
            plot_pointwise_error_curves(
                curves=pointwise_curves,
                mode=mode,
                title=f"{mode.title()}",
                output_path=os.path.join(output_dir, f"pointwise_error_{dim}d_{mode.title()}_tau{pointwise_tau_final}_curves{pointwise_num_curves}_ntu{n_tu}_nx{n_x}.png"),
                logy=False,
            )

    summary = {
        "spatial_dim": int(dim),
        "mode": mode,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "num_f0_values": int(num_f0s),
        "num_tau_increments": int(nsteps),
        "num_tau_checkpoints": int(len(history)),
        "num_t_values": int(len(grid["t"])),
        "num_x_values_per_dim": int(len(grid["x_axes"][0])),
        "num_spatial_points": int(np.prod(grid["spatial_shape"])),
        "plot_kind": plot_kind,
        "tau_final_requested": float(tau_final),
        "tau_final_simulated": float(history[-1, 0]) if need_history else float("nan"),
        "dtau": float(dtau),
        "pointwise_tau_final_requested": float(pointwise_tau_final),
        "pointwise_dtau": float(pointwise_dtau),
        "pointwise_num_curves": int(pointwise_num_curves),
        "eps": float(eps),
        "final_l2_error": float(history[-1, 1]) if need_history else float("nan"),
        "final_linf_error": float(history[-1, 2]) if need_history else float("nan"),
        "solution_scatter_pearsonr": float(scatter_pearsonr),
        "pointwise_curves": [
            {
                "curve_index": int(curve["curve_index"]),
                "f0_index": int(curve["f0_index"]),
                "t_value": float(curve["t_value"]),
                "u_value": float(curve["u_value"]),
                "x_value": [float(v) for v in curve["x_value"]],
                "num_tau_values": int(len(curve["tau_values"])),
                "final_numerical_error": float(curve["numerical_error"][-1]),
                "final_predicted_error": float(curve["predicted_error"][-1]),
                "max_abs_curve_gap": float(
                    np.max(np.abs(np.asarray(curve["numerical_error"]) - np.asarray(curve["predicted_error"])))
                ),
            }
            for curve in pointwise_curves
        ],
    }
    if output_dir is not None:
        with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    if f_final is None:
        # Pointwise-only runs intentionally skip the full-grid checkpoint path.
        f_final = torch.empty(0, device=device, dtype=dtype)
    return f_final, history, grid, velocity, summary, scatter_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch direct grid simulation of functional gradient-flow PDEs.")
    parser.add_argument("--plot_kind", type=str, choices=["scatter", "pointwise_error", "both"], default="scatter")
    parser.add_argument("--spatial_dim", type=int, choices=[1, 2, 3, 4, 5], default=1)
    parser.add_argument("--mode", type=str, choices=["eulerian", "lagrangian", "both"], default="both")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, choices=["float32", "float64"], default="float64")
    parser.add_argument("--n_tu", type=int, default=None)
    parser.add_argument("--n_x", type=int, default=None)
    parser.add_argument("--tau_final", type=float, default=0.20)
    parser.add_argument("--dtau", type=float, default=None)
    parser.add_argument("--eps", type=float, default=0.15)
    parser.add_argument("--num_f0s", type=int, default=1)
    parser.add_argument("--checkpoints", type=int, default=20)
    parser.add_argument("--pointwise_tau_final", type=float, default=None)
    parser.add_argument("--pointwise_dtau", type=float, default=None)
    parser.add_argument("--pointwise_num_curves", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    root_output_dir = args.output_dir
    if root_output_dir is None:
        root_output_dir = os.path.join(os.path.dirname(__file__), "outputs")

    modes = ["eulerian", "lagrangian"] if args.mode == "both" else [args.mode]
    run_output_dir = os.path.join(root_output_dir, f"{args.spatial_dim}d")
    os.makedirs(run_output_dir, exist_ok=True)

    scatter_by_mode: Dict[str, Dict[str, object]] = {}
    for mode in modes:
        mode_output_dir = os.path.join(run_output_dir, mode)
        os.makedirs(mode_output_dir, exist_ok=True)
        _, history, _, _, summary, scatter_data = run_demo(
            dim=args.spatial_dim,
            mode=mode,
            device=device,
            dtype=dtype,
            n_tu=args.n_tu,
            n_x=args.n_x,
            tau_final=args.tau_final,
            dtau=args.dtau,
            eps=args.eps,
            num_f0s=args.num_f0s,
            checkpoints=args.checkpoints,
            plot_kind=args.plot_kind,
            pointwise_tau_final=args.pointwise_tau_final,
            pointwise_dtau=args.pointwise_dtau,
            pointwise_num_curves=args.pointwise_num_curves,
            output_dir=mode_output_dir,
        )
        if scatter_data is not None:
            scatter_by_mode[mode] = scatter_data
        print(f"{args.spatial_dim}D {mode.title()} ({device.type}, {args.dtype}):")
        print("  final tau        =", history[-1, 0])
        print("  final L2 error   =", history[-1, 1])
        print("  final Linf error =", history[-1, 2])
        print("  num f0 values    =", summary["num_f0_values"])
        print("  tau increments   =", summary["num_tau_increments"])
        print("  dtau             =", summary["dtau"])
        print("  num t/u values   =", summary["num_t_values"])
        print("  num x values     =", summary["num_spatial_points"])
        print("  scatter pearsonr =", summary["solution_scatter_pearsonr"])
        print("  pointwise curves =", summary["pointwise_num_curves"])
        print("  pointwise tau    =", summary["pointwise_tau_final_requested"])
        print(f"Saved figures to: {mode_output_dir}")

    if args.plot_kind in {"scatter", "both"} and len(scatter_by_mode) > 1:
        plot_combined_solution_scatter(
            scatter_by_mode=scatter_by_mode,
            title=f"Functional gradient-flow solution scatter comparison | dim={args.spatial_dim}",
            output_path=os.path.join(run_output_dir, f"solution_scatter_all_modes_{args.spatial_dim}d_{args.num_f0s}nf0s.png"),
        )
