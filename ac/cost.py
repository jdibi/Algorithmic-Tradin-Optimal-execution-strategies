"""ac.cost
=======================
Expected implementation shortfall (mean) and risk (variance) for
Almgren–Chriss schedules, plus an efficient frontier helper.

This module is schedule-agnostic: it can compute cost metrics from arrays
`(t, x, v)` (continuous-time grid) or `(t_k, x_k, v_k)` (discrete slices),
and also provides convenience wrappers that *build* the AC optimal schedules
(via :mod:`ac.model`) before evaluating costs.

Formulas (baseline AC model)
----------------------------
- Expected cost (implementation shortfall):
    E[C] = (γ/2) X0^2 + η ∫ v(t)^2 dt
  or, in discrete time with slice size v_k over Δt:
    E[C] = (γ/2) X0^2 + η Σ (v_k^2 / Δt)

- Variance of cost (price risk from diffusion):
    Var[C] = σ^2 ∫ x(t)^2 dt
  or, in discrete time (inventory held during each interval):
    Var[C] = σ^2 Σ x_k^2 Δt

The mean–variance objective is then:  J = E[C] + λ Var[C].

References
----------
- Almgren, R. & Chriss, N. (2001). Optimal execution of portfolio transactions.
  Journal of Risk, 3(2), 5–39.
- Almgren, R. (2003). Optimal execution with nonlinear impact functions and
  trading-enhanced risk. Applied Mathematical Finance, 10(1), 1–18.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .model import (
    continuous_schedule,
    discrete_schedule,
    kappa as kappa_fn,
)

__all__ = [
    "CostBreakdown",
    "expected_variance_from_continuous",
    "expected_variance_from_discrete",
    "continuous_cost",
    "discrete_cost",
    "efficient_frontier",
]


# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class CostBreakdown:
    """Container for cost metrics.

    Attributes
    ----------
    E_cost : float
        Expected implementation shortfall E[C].
    Var_cost : float
        Variance Var[C].
    objective : float
        Mean–variance objective J = E_cost + lam * Var_cost.
    extras : dict
        Extra derived quantities (e.g., kappa, dt).
    """

    E_cost: float
    Var_cost: float
    objective: float
    extras: Dict[str, float]


# -----------------------------
# Core calculators (from arrays)
# -----------------------------

def expected_variance_from_continuous(
    *,
    t: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    X0: float,
    sigma: float,
    eta: float,
    gamma: float = 0.0,
) -> Tuple[float, float]:
    """Compute (E[C], Var[C]) from continuous grids.

    Parameters
    ----------
    t, x, v : ndarray
        Time grid, shares remaining, and trading rate arrays (same length).
    X0, sigma, eta, gamma : float
        Model parameters.

    Returns
    -------
    (E_cost, Var_cost)
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    if not (t.ndim == x.ndim == v.ndim == 1) or not (len(t) == len(x) == len(v)):
        raise ValueError("t, x, v must be 1-D arrays of equal length")

    # Expected cost: (γ/2) X0^2 + η ∫ v(t)^2 dt
    temp_cost = eta * np.trapz(v ** 2, t)
    perm_cost = 0.5 * gamma * (X0 ** 2)
    E_cost = perm_cost + temp_cost

    # Variance: σ^2 ∫ x(t)^2 dt
    Var_cost = (sigma ** 2) * np.trapz(x ** 2, t)
    return float(E_cost), float(Var_cost)


def expected_variance_from_discrete(
    *,
    t_k: np.ndarray,
    x_k: np.ndarray,
    v_k: np.ndarray,
    X0: float,
    sigma: float,
    eta: float,
    gamma: float = 0.0,
) -> Tuple[float, float]:
    """Compute (E[C], Var[C]) from discrete boundaries/slices.

    Parameters
    ----------
    t_k : ndarray, shape (N+1,)
        Grid boundaries in [0, T]. Must be strictly increasing.
    x_k : ndarray, shape (N+1,)
        Inventory at boundaries (shares remaining).
    v_k : ndarray, shape (N,)
        Slice sizes per interval: v_k = x_k - x_{k+1} (shares, not a rate).
    X0, sigma, eta, gamma : float
        Model parameters.

    Returns
    -------
    (E_cost, Var_cost)
    """
    t_k = np.asarray(t_k, dtype=float)
    x_k = np.asarray(x_k, dtype=float)
    v_k = np.asarray(v_k, dtype=float)

    if not (t_k.ndim == x_k.ndim == 1 and v_k.ndim == 1):
        raise ValueError("t_k, x_k, v_k must be 1-D arrays")
    if not (len(t_k) == len(x_k) and len(v_k) == len(t_k) - 1):
        raise ValueError("Shapes must satisfy len(t_k)=len(x_k)=N+1 and len(v_k)=N")

    dt = np.diff(t_k)
    if not np.all(dt > 0):
        raise ValueError("t_k must be strictly increasing")

    # For uniform grids, dt is constant; but support non-uniform for robustness
    # Expected temp cost: η Σ ( (v_k / dt_k)^2 * dt_k ) = η Σ (v_k^2 / dt_k )
    temp_cost = float(np.sum(eta * (v_k ** 2) / dt))
    perm_cost = 0.5 * gamma * (X0 ** 2)
    E_cost = perm_cost + temp_cost

    # Variance: σ^2 Σ x_k^2 * dt_k, using *left* boundary inventory within interval
    Var_cost = float((sigma ** 2) * np.sum((x_k[:-1] ** 2) * dt))

    return E_cost, Var_cost


# -----------------------------
# Convenience wrappers (build schedules)
# -----------------------------

def continuous_cost(
    *,
    X0: float,
    T: float,
    sigma: float,
    eta: float,
    lam: float,
    gamma: float = 0.0,
    N: int = 1001,
    t_grid: Optional[Iterable[float]] = None,
    return_schedule: bool = False,
) -> Tuple[CostBreakdown, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Compute cost metrics for the AC *continuous* optimal schedule.

    Returns a :class:`CostBreakdown` and, if `return_schedule=True`, also
    the `(t, x, v)` arrays used for integration.
    """
    t, x, v, params = continuous_schedule(
        X0=X0, T=T, sigma=sigma, eta=eta, lam=lam, gamma=gamma, N=N, t_grid=t_grid
    )
    E_cost, Var_cost = expected_variance_from_continuous(
        t=t, x=x, v=v, X0=X0, sigma=sigma, eta=eta, gamma=gamma
    )
    J = E_cost + lam * Var_cost
    extras = {"kappa": params.get("kappa", np.nan), "T": float(T)}
    out = CostBreakdown(E_cost=E_cost, Var_cost=Var_cost, objective=J, extras=extras)
    if return_schedule:
        return out, (t, x, v)
    return out, None


def discrete_cost(
    *,
    X0: float,
    T: float,
    sigma: float,
    eta: float,
    lam: float,
    N: int,
    gamma: float = 0.0,
    return_schedule: bool = False,
) -> Tuple[CostBreakdown, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Compute cost metrics for the AC *discrete* optimal schedule.

    Returns a :class:`CostBreakdown` and, if `return_schedule=True`, also
    the `(t_k, x_k, v_k)` arrays used for evaluation.
    """
    t_k, x_k, v_k, params = discrete_schedule(
        X0=X0, T=T, sigma=sigma, eta=eta, lam=lam, N=N, gamma=gamma
    )
    E_cost, Var_cost = expected_variance_from_discrete(
        t_k=t_k, x_k=x_k, v_k=v_k, X0=X0, sigma=sigma, eta=eta, gamma=gamma
    )
    J = E_cost + lam * Var_cost
    extras = {"kappa": params.get("kappa", np.nan), "dt": params.get("dt", np.nan), "T": float(T)}
    out = CostBreakdown(E_cost=E_cost, Var_cost=Var_cost, objective=J, extras=extras)
    if return_schedule:
        return out, (t_k, x_k, v_k)
    return out, None


# -----------------------------
# Efficient frontier helper
# -----------------------------

def efficient_frontier(
    X0: float,
    T: float,
    sigma: float,
    eta: float,
    gamma: float,
    lambdas: Sequence[float],
    *,
    grid: str = "continuous",
    N: int = 241,
) -> List[Dict[str, float]]:
    """Compute (E, Var) along a sweep of λ values.

    Parameters
    ----------
    X0, T, sigma, eta, gamma : float
        Model parameters.
    lambdas : sequence of float
        Risk-aversion values to evaluate.
    grid : {"continuous", "discrete"}, default "continuous"
        Which schedule to use when generating the frontier.
    N : int, default 241
        Resolution of the grid (points for continuous; slices for discrete).

    Returns
    -------
    list of dict
        Each dict contains {"lambda", "E_cost", "Var_cost", "objective", "kappa"}.
    """
    results: List[Dict[str, float]] = []

    for lam in lambdas:
        if grid == "continuous":
            cb, _ = continuous_cost(
                X0=X0, T=T, sigma=sigma, eta=eta, lam=lam, gamma=gamma, N=N, return_schedule=False
            )
        elif grid == "discrete":
            cb, _ = discrete_cost(
                X0=X0, T=T, sigma=sigma, eta=eta, lam=lam, gamma=gamma, N=N, return_schedule=False
            )
        else:
            raise ValueError("grid must be 'continuous' or 'discrete'")

        results.append({
            "lambda": float(lam),
            "E_cost": float(cb.E_cost),
            "Var_cost": float(cb.Var_cost),
            "objective": float(cb.objective),
            "kappa": float(cb.extras.get("kappa", np.nan)),
        })

    # Sort by lambda for neatness
    results.sort(key=lambda d: d["lambda"])
    return results


# End of module

