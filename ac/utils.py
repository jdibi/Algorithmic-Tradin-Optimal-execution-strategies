"""ac.utils
=================
Small utility helpers for the Almgren–Chriss repository.

These functions are intentionally lightweight and dependency-free (beyond NumPy)
so they can be reused across modules (model, schedules, cost, tests).

Highlights
----------
- Validation helpers for parameters and schedule shapes.
- Time grid builders.
- Inventory/slice conversions.
- Volatility scaling across horizons.
- Tiny numerics conveniences (near-zero checks, clipping).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    # Validation
    "require_positive",
    "require_nonnegative",
    "is_strictly_increasing",
    "validate_continuous_schedule",
    "validate_discrete_schedule",
    # Grids
    "uniform_time_grid",
    # Inventory / slices
    "slices_from_inventory",
    "inventory_from_slices",
    # Volatility scaling
    "scale_vol",
    # Numerics
    "near_zero",
    "clip_small",
    # Reproducibility
    "temp_seed",
]


# -----------------------------
# Validation helpers
# -----------------------------

def require_positive(name: str, val: float) -> None:
    """Raise ValueError if ``val`` is not strictly > 0."""
    if not (val > 0):
        raise ValueError(f"{name} must be > 0; got {val!r}")


def require_nonnegative(name: str, val: float) -> None:
    """Raise ValueError if ``val`` is negative."""
    if not (val >= 0):
        raise ValueError(f"{name} must be >= 0; got {val!r}")


def is_strictly_increasing(a: Iterable[float]) -> bool:
    """Return True if the 1-D array is strictly increasing."""
    arr = np.asarray(list(a), dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        return False
    return bool(np.all(np.diff(arr) > 0))


def validate_continuous_schedule(t: np.ndarray, x: np.ndarray, v: np.ndarray) -> None:
    """Validate shapes for (t, x, v) continuous arrays."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    if not (t.ndim == x.ndim == v.ndim == 1):
        raise ValueError("t, x, v must be 1-D arrays")
    if not (len(t) == len(x) == len(v)):
        raise ValueError("t, x, v must have equal length")
    if not is_strictly_increasing(t):
        raise ValueError("t must be strictly increasing")


def validate_discrete_schedule(t_k: np.ndarray, x_k: np.ndarray, v_k: np.ndarray) -> None:
    """Validate shapes for (t_k, x_k, v_k) discrete arrays."""
    t_k = np.asarray(t_k, dtype=float)
    x_k = np.asarray(x_k, dtype=float)
    v_k = np.asarray(v_k, dtype=float)
    if not (t_k.ndim == x_k.ndim == 1 and v_k.ndim == 1):
        raise ValueError("t_k, x_k, v_k must be 1-D arrays")
    if not (len(t_k) == len(x_k) and len(v_k) == len(t_k) - 1):
        raise ValueError("Shapes must satisfy len(t_k)=len(x_k)=N+1 and len(v_k)=N")
    if not is_strictly_increasing(t_k):
        raise ValueError("t_k must be strictly increasing")


# -----------------------------
# Time grids
# -----------------------------

def uniform_time_grid(T: float, N: int, *, t0: float = 0.0, endpoint: bool = True) -> np.ndarray:
    """Uniform time grid from ``t0`` to ``t0+T`` with ``N`` intervals.

    If ``endpoint`` is True, returns N+1 points (boundaries). If False, returns N points (centers).
    """
    require_positive("T", T)
    require_positive("N", N)
    a, b = float(t0), float(t0 + T)
    if endpoint:
        return np.linspace(a, b, int(N) + 1)
    else:
        return np.linspace(a + T / (2 * N), b - T / (2 * N), int(N))


# -----------------------------
# Inventory / slices conversions
# -----------------------------

def slices_from_inventory(x_k: np.ndarray) -> np.ndarray:
    """Return slice sizes v_k from boundary inventory x_k (v_k = x_k - x_{k+1})."""
    x_k = np.asarray(x_k, dtype=float)
    if x_k.ndim != 1 or x_k.size < 2:
        raise ValueError("x_k must be 1-D with length >= 2")
    return x_k[:-1] - x_k[1:]


def inventory_from_slices(v_k: np.ndarray, X0: float) -> np.ndarray:
    """Return boundary inventory x_k from slice sizes v_k and initial X0.

    The last element is forced to 0 to remove accumulated floating error.
    """
    v_k = np.asarray(v_k, dtype=float)
    x_k = np.empty(v_k.size + 1, dtype=float)
    x_k[0] = float(X0)
    x_k[1:] = X0 - np.cumsum(v_k)
    x_k[-1] = 0.0
    return x_k


# -----------------------------
# Volatility scaling
# -----------------------------

def scale_vol(vol: float, *, from_horizon: float, to_horizon: float) -> float:
    """Scale a diffusion volatility between horizons via sqrt-time rule.

    ``vol`` is the standard deviation over ``from_horizon``. Returns the
    standard deviation over ``to_horizon``.
    """
    require_positive("from_horizon", from_horizon)
    require_positive("to_horizon", to_horizon)
    return float(vol) * np.sqrt(to_horizon / from_horizon)


# -----------------------------
# Numerics
# -----------------------------

def near_zero(x: float, *, tol: float = 1e-12) -> bool:
    """Return True if abs(x) <= tol."""
    return bool(abs(float(x)) <= tol)


def clip_small(a: np.ndarray, *, tol: float = 1e-12) -> np.ndarray:
    """Zero-out entries with absolute value <= tol (returns a copy)."""
    arr = np.asarray(a, dtype=float).copy()
    arr[np.abs(arr) <= tol] = 0.0
    return arr


# -----------------------------
# Reproducibility
# -----------------------------
@contextmanager
def temp_seed(seed: Optional[int]):
    """Temporarily set NumPy's RNG seed within a ``with`` block.

    Example
    -------
    >>> import numpy as np
    >>> rng_before = np.random.random()
    >>> with temp_seed(123):
    ...     x = np.random.random(3)
    >>> rng_after = np.random.random()
    """
    if seed is None:
        yield
        return
    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        yield
    finally:
        np.random.set_state(state)


# End of module

