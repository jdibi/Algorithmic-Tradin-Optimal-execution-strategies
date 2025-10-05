"""ac.model
=======================
Core Almgren–Chriss (AC) optimal execution schedules (continuous and discrete time).

This module provides numerically stable implementations of the closed-form
Almgren–Chriss optimal trajectory `x*(t)` and trading rate `v*(t)` under
linear temporary impact and mean–variance risk penalization.

References
----------
- Almgren, R. & Chriss, N. (2001). Optimal execution of portfolio transactions.
  Journal of Risk, 3(2), 5–39.
- Almgren, R. (2003). Optimal execution with nonlinear impact functions and
  trading-enhanced risk. Applied Mathematical Finance, 10(1), 1–18.

Notes
-----
- In the basic AC setup with constant coefficients, the shape of the optimal
  trajectory depends on `kappa = sqrt(lambda * sigma^2 / eta)` and does not
  depend on permanent impact `gamma`. Permanent impact contributes to expected
  cost but not to the trajectory under these assumptions.
- These routines handle the `lambda → 0` (risk-neutral) limit by falling back to
  a TWAP schedule to avoid 0/0 numerical issues in the hyperbolic functions.

Example
-------
>>> from ac.model import continuous_schedule
>>> t, x, v, params = continuous_schedule(
...     X0=1_000_000, T=1.0, sigma=0.02, eta=1e-6, lam=5e-6
... )
>>> float(params["kappa"]) > 0
True
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional, Tuple, Dict

import numpy as np

__all__ = [
    "ScheduleParams",
    "kappa",
    "continuous_schedule",
    "discrete_schedule",
    "twap_schedule",
]


# -----------------------------
# Dataclasses & helpers
# -----------------------------

@dataclass(frozen=True)
class ScheduleParams:\n    """Parameters that define an AC schedule.

    Attributes
    ----------
    X0 : float
        Initial position (shares). Positive for sell, negative for buy.
    T : float
        Execution horizon (in the same time units used for sigma).
    sigma : float
        Price diffusion volatility per sqrt(time) (e.g., daily vol).
    eta : float
        Temporary impact coefficient.
    lam : float
        Risk aversion (mean–variance penalty weight, >= 0).
    gamma : float
        Permanent impact coefficient (does not change the *shape* here).
    N : int | None
        Optional number of grid points/slices used to return the schedule.
    """

    X0: float
    T: float
    sigma: float
    eta: float
    lam: float
    gamma: float = 0.0
    N: Optional[int] = None



_EPS = 1e-15
_SMALL_KAPPA_THRESHOLD = 1e-8  # threshold on kappa * T for TWAP fallback


def _validate_positive(name: str, val: float, strictly: bool = True) -> None:
    if strictly and not (val > 0):
        raise ValueError(f"{name} must be > 0; got {val!r}")
    if not strictly and not (val >= 0):
        raise ValueError(f"{name} must be >= 0; got {val!r}")


def kappa(lam: float, sigma: float, eta: float) -> float:
    """Compute kappa = sqrt(lambda * sigma^2 / eta) with safeguards.

    Returns 0.0 when lam == 0 to support the risk-neutral limit (TWAP).
    """
    _validate_positive("eta", eta)
    _validate_positive("sigma", sigma)
    _validate_positive("lam", lam, strictly=False)

    if lam == 0:
        return 0.0
    return math.sqrt((lam * sigma * sigma) / eta)


# -----------------------------
# Public API
# -----------------------------

def continuous_schedule(
    *,
    X0: float,
    T: float,
    sigma: float,
    eta: float,
    lam: float,
    gamma: float = 0.0,
    N: int = 1001,
    t_grid: Optional[Iterable[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Closed-form continuous-time AC optimal schedule.

    Parameters
    ----------
    X0 : float
        Initial position (shares). Positive for sell, negative for buy.
    T : float
        Total time horizon (e.g., days). Must be > 0.
    sigma : float
        Volatility per sqrt(time).
    eta : float
        Temporary impact coefficient (> 0).
    lam : float
        Risk aversion (>= 0). Controls front-loading via kappa.
    gamma : float, default 0.0
        Permanent impact coefficient (does not affect shape here).
    N : int, default 1001
        Number of points in the uniform time grid if `t_grid` is not provided.
    t_grid : Iterable[float], optional
        Custom time grid in [0, T]. If provided, overrides `N`.

    Returns
    -------
    t : ndarray of shape (M,)
        Time grid in [0, T].
    x : ndarray of shape (M,)
        Optimal *shares remaining* at each t.
    v : ndarray of shape (M,)
        Optimal *trading rate* v(t) = -dx/dt (shares per unit time).
    params : dict
        Dictionary with derived quantities: {"kappa": float}.

    Notes
    -----
    For kappa -> 0 (risk-neutral), this function returns a TWAP trajectory
    with x(t) = X0 * (1 - t/T) and constant v(t) = X0 / T.
    """
    _validate_positive("T", T)
    _validate_positive("sigma", sigma)
    _validate_positive("eta", eta)
    _validate_positive("lam", lam, strictly=False)

    k = kappa(lam, sigma, eta)

    if t_grid is None:
        _validate_positive("N", N)
        t = np.linspace(0.0, T, int(N))
    else:
        t = np.asarray(list(t_grid), dtype=float)
        if t.ndim != 1:
            raise ValueError("t_grid must be 1-D")
        if (t[0] < -_EPS) or (t[-1] > T + _EPS):
            raise ValueError("t_grid must lie within [0, T]")

    # Risk-neutral (kappa ~ 0) → TWAP to avoid 0/0 in sinh(κT)
    if k * T < _SMALL_KAPPA_THRESHOLD:
        x = X0 * (1.0 - t / T)
        v = np.full_like(t, fill_value=X0 / T)
        return t, x, v, {"kappa": k}

    # Optimal closed form
    denom = math.sinh(k * T)
    # Avoid catastrophic division if denom is tiny (already handled above)
    inv_denom = 1.0 / denom

    # x(t) = X0 * sinh(k (T - t)) / sinh(k T)
    s = np.sinh(k * (T - t)) * inv_denom
    x = X0 * s

    # v(t) = k X0 * cosh(k (T - t)) / sinh(k T)
    c = np.cosh(k * (T - t)) * inv_denom
    v = (k * X0) * c

    return t, x, v, {"kappa": k}


def twap_schedule(*, X0: float, T: float, N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience TWAP schedule (risk-neutral baseline).

    Returns a uniform grid t_k (k=0..N), boundary inventory x_k, and slice sizes v_k.
    """
    _validate_positive("T", T)
    _validate_positive("N", N)
    dt = T / N
    t = np.linspace(0.0, T, N + 1)
    x = X0 * (1.0 - t / T)
    # Boundary inventory: enforce exact endpoints
    x[0] = X0
    x[-1] = 0.0
    v = np.full(N, X0 / N)
    return t, x, v


def discrete_schedule(
    *,
    X0: float,
    T: float,
    sigma: float,
    eta: float,
    lam: float,
    N: int,
    gamma: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Closed-form discrete-time AC optimal schedule.

    We return inventory on *boundaries* (k=0..N) and slice sizes for each
    interval (k=0..N-1). The discrete schedule mirrors the continuous solution
    and converges to it as N increases.

    Parameters
    ----------
    X0, T, sigma, eta, lam, gamma : see ``continuous_schedule``.
    N : int
        Number of equal-width time slices.

    Returns
    -------
    t_k : ndarray, shape (N+1,)
        Grid boundaries in [0, T].
    x_k : ndarray, shape (N+1,)
        Optimal *shares remaining* at boundaries.
    v_k : ndarray, shape (N,)
        Optimal slice sizes for each interval: v_k = x_k - x_{k+1}.
    params : dict
        {"kappa": float, "dt": float}
    """
    _validate_positive("T", T)
    _validate_positive("sigma", sigma)
    _validate_positive("eta", eta)
    _validate_positive("lam", lam, strictly=False)
    _validate_positive("N", N)

    dt = T / N
    t_k = np.linspace(0.0, T, N + 1)
    k = kappa(lam, sigma, eta)

    # Risk-neutral / small-kappa limit → TWAP
    if k * T < _SMALL_KAPPA_THRESHOLD:
        t, x, v = twap_schedule(X0=X0, T=T, N=N)
        return t, x, v, {"kappa": k, "dt": dt}

    # Discrete analogue uses the same sinh structure evaluated on the grid
    # x_k = X0 * sinh(kappa * dt * (N - k)) / sinh(kappa * dt * N)
    z = k * dt
    denom = math.sinh(z * N)
    inv_denom = 1.0 / denom

    idx = np.arange(N + 1)
    x_k = X0 * np.sinh(z * (N - idx)) * inv_denom

    # Enforce exact boundary values to avoid tiny numerical residue
    x_k[0] = X0
    x_k[-1] = 0.0

    v_k = x_k[:-1] - x_k[1:]

    return t_k, x_k, v_k, {"kappa": k, "dt": dt}


# End of module

