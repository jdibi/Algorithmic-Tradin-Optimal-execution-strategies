"""ac.schedules
=========================
Helper utilities to build, transform, and validate execution schedules.

This module complements :mod:`ac.model` with convenience wrappers and
utility functions for working with *continuous* (t, x(t), v(t)) and
*discrete* (t_k, x_k, v_k) schedules, as well as basic heuristic
schedules like TWAP/POV and resampling helpers.

Notation
--------
- Continuous: t ∈ [0, T], inventory x(t) (shares remaining), rate v(t) = -dx/dt.
- Discrete: boundaries t_k (k=0..N), inventory x_k at boundaries, slice sizes
  v_k = x_k - x_{k+1} (shares executed during (t_k, t_{k+1}]).

All helpers are sign-agnostic: X0>0 means sell; X0<0 means buy. Slices will
sum to X0 (with sign) and inventories go from x_0=X0 to x_N=0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

import numpy as np

from .model import (
    continuous_schedule as ac_continuous_schedule,
    discrete_schedule as ac_discrete_schedule,
    twap_schedule as ac_twap_schedule,
)

__all__ = [
    # Dataclasses
    "ContinuousSchedule",
    "DiscreteSchedule",
    # Builders / wrappers
    "build_ac_continuous",
    "build_ac_discrete",
    "build_twap",
    "build_pov",
    # Profiles
    "u_shape_profile",
    "gaussian_profile",
    # Resampling / transforms
    "discrete_from_continuous",
    "continuous_from_discrete",
    # Validation
    "check_monotone_inventory",
]


# -----------------------------
# Dataclasses
# -----------------------------

@dataclass(frozen=True)
class ContinuousSchedule:
    t: np.ndarray  # shape (M,)
    x: np.ndarray  # shape (M,)
    v: np.ndarray  # shape (M,)


@dataclass(frozen=True)
class DiscreteSchedule:
    t_k: np.ndarray  # shape (N+1,)
    x_k: np.ndarray  # shape (N+1,)
    v_k: np.ndarray  # shape (N,)


# -----------------------------
# Builders / wrappers
# -----------------------------

def build_ac_continuous(
    *,
    X0: float,
    T: float,
    sigma: float,
    eta: float,
    lam: float,
    gamma: float = 0.0,
    N: int = 1001,
    t_grid: Optional[Iterable[float]] = None,
) -> Tuple[ContinuousSchedule, dict]:
    """Wrapper around :func:`ac.model.continuous_schedule` returning a dataclass.

    Returns (ContinuousSchedule, params_dict).
    """
    t, x, v, params = ac_continuous_schedule(
        X0=X0, T=T, sigma=sigma, eta=eta, lam=lam, gamma=gamma, N=N, t_grid=t_grid
    )
    return ContinuousSchedule(t=t, x=x, v=v), params


def build_ac_discrete(
    *,
    X0: float,
    T: float,
    sigma: float,
    eta: float,
    lam: float,
    N: int,
    gamma: float = 0.0,
) -> Tuple[DiscreteSchedule, dict]:
    """Wrapper around :func:`ac.model.discrete_schedule` returning a dataclass.

    Returns (DiscreteSchedule, params_dict).
    """
    t_k, x_k, v_k, params = ac_discrete_schedule(
        X0=X0, T=T, sigma=sigma, eta=eta, lam=lam, N=N, gamma=gamma
    )
    return DiscreteSchedule(t_k=t_k, x_k=x_k, v_k=v_k), params


def build_twap(*, X0: float, T: float, N: int) -> DiscreteSchedule:
    """Simple TWAP schedule on a uniform grid.

    For sells (X0>0), v_k >= 0 and sum(v_k)=X0. For buys, signs flip.
    """
    sign = 1.0 if X0 >= 0 else -1.0
    t_k, x_k, v_k = ac_twap_schedule(X0=abs(X0), T=T, N=N)
    return DiscreteSchedule(t_k=t_k, x_k=sign * x_k, v_k=sign * v_k)


# -----------------------------
# Profiles & POV schedule
# -----------------------------

def _normalize_nonnegative(weights: np.ndarray) -> np.ndarray:
    w = np.maximum(0.0, np.asarray(weights, dtype=float))
    s = float(w.sum())
    if s <= 0:
        raise ValueError("profile weights must have positive sum")
    return w / s


def u_shape_profile(N: int, alpha: float = 1.5) -> np.ndarray:
    """U-shaped intraday profile (peaks at open/close).

    Uses a Beta(alpha, alpha) density sampled on N bins and mirrored to peak at
    both ends. alpha in (0, inf): larger → more extreme U-shape.
    """
    if N <= 0:
        raise ValueError("N must be positive")
    k = np.arange(N) + 0.5
    u = k / N
    # Symmetric U-shape: f(u) ∝ u^(alpha-1) (1-u)^(alpha-1)
    w = (np.maximum(u, 1 - u) ** (alpha - 1)) * (np.minimum(u, 1 - u) ** (alpha - 1))
    # This formula peaks in the middle; invert to get U-shape
    w = 1.0 - (w / w.max())
    return _normalize_nonnegative(w)


def gaussian_profile(N: int, center: float = 0.5, width: float = 0.15) -> np.ndarray:
    """Simple Gaussian-like profile across N slices (not normalized)."""
    if N <= 0:
        raise ValueError("N must be positive")
    k = np.arange(N) + 0.5
    u = k / N
    w = np.exp(-0.5 * ((u - center) / max(width, 1e-6)) ** 2)
    return _normalize_nonnegative(w)


def build_pov(*, X0: float, T: float, weights: Iterable[float]) -> DiscreteSchedule:
    """Participation-of-Volume-style schedule from a nonnegative weight profile.

    Parameters
    ----------
    X0 : float
        Shares to execute (signed). Positive sell, negative buy.
    T : float
        Horizon.
    weights : Iterable[float]
        Nonnegative slice weights; will be normalized to sum to 1.

    Returns
    -------
    DiscreteSchedule
        With uniform time grid and v_k proportional to weights.
    """
    w = _normalize_nonnegative(np.asarray(list(weights), dtype=float))
    N = int(w.size)
    t_k = np.linspace(0.0, T, N + 1)
    v_k = X0 * w  # signed
    # Build boundary inventory
    x_k = np.empty(N + 1, dtype=float)
    x_k[0] = X0
    x_k[1:] = X0 - np.cumsum(v_k)
    # Numerical safety: enforce terminal
    x_k[-1] = 0.0
    return DiscreteSchedule(t_k=t_k, x_k=x_k, v_k=v_k)


# -----------------------------
# Resampling / transforms
# -----------------------------

def discrete_from_continuous(cs: ContinuousSchedule, N: int) -> DiscreteSchedule:
    """Sample a continuous schedule onto N uniform slices.

    Uses left-boundary inventory per interval and slice sizes from differences.
    """
    if N <= 0:
        raise ValueError("N must be positive")
    t_k = np.linspace(float(cs.t[0]), float(cs.t[-1]), N + 1)
    # Interpolate inventory at boundaries
    x_k = np.interp(t_k, cs.t, cs.x)
    v_k = x_k[:-1] - x_k[1:]
    # Enforce exact endpoints
    x_k[0] = cs.x[0]
    x_k[-1] = cs.x[-1]
    return DiscreteSchedule(t_k=t_k, x_k=x_k, v_k=v_k)


def continuous_from_discrete(ds: DiscreteSchedule, M: int = 1001) -> ContinuousSchedule:
    """Piecewise-linear interpolation of a discrete schedule onto M points.

    The trading rate v(t) is piecewise-constant within each interval.
    """
    if M <= 2:
        raise ValueError("M must be > 2")
    t0, T = float(ds.t_k[0]), float(ds.t_k[-1])
    t = np.linspace(t0, T, M)
    # Linear interpolate inventory; rate is piecewise constant per interval
    x = np.interp(t, ds.t_k, ds.x_k)
    dt = np.diff(ds.t_k)
    rate = (ds.x_k[:-1] - ds.x_k[1:]) / dt  # equals v_k / dt
    # Assign rate per interval to interior sample points
    idx = np.minimum(np.searchsorted(ds.t_k[1:], t, side="right"), len(rate) - 1)
    v = rate[idx]
    return ContinuousSchedule(t=t, x=x, v=v)


# -----------------------------
# Validation helpers
# -----------------------------

def check_monotone_inventory(ds: DiscreteSchedule, *, tol: float = 1e-8) -> bool:
    """Return True if inventory is monotone towards zero with small tolerance.

    Works for both sells (x_k decreasing to 0) and buys (x_k increasing to 0).
    """
    x0 = ds.x_k[0]
    if x0 == 0:
        return bool(abs(ds.x_k[-1]) <= tol)
    if x0 > 0:  # sell
        return bool(np.all(np.diff(ds.x_k) <= tol) and abs(ds.x_k[-1]) <= tol)
    else:  # buy
        return bool(np.all(np.diff(ds.x_k) >= -tol) and abs(ds.x_k[-1]) <= tol)


# End of module

