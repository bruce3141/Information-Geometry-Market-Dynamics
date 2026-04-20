"""
p3_1.phase_space

Shared phase-space construction for the SDHO regression. Implements the
fencepost-corrected alignment used throughout P3-1 and consistent with
[1] (Dean, 2025) and [2] (Dean, 2026).

The previous-session debugging history that led to the current alignment:
an earlier version of this code paired y_t (the increment) with the END
position rather than the START position, which introduced a fencepost
error that shifted the fitted Omega by approximately k. The correct
alignment, documented in build_phase_space() below, drops the LAST
element of x_full so that y[t] = x_full[t+1] - x_full[t] is paired with
x[t] = x_full[t]: the increment starts FROM x[t]. Getting this wrong
silently degrades the fit and breaks the SPY 1993-2025 sanity check
against [2] Table 7a.
"""
from __future__ import annotations

import numpy as np

LOOKBACK_DEFAULT = 5  # baseline n=5 daily, matching [1] and [2]


def build_phase_space(
    prices: np.ndarray,
    n: int = LOOKBACK_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (x, y, dy) phase-space arrays for the SDHO regression.

    Conventions, matching [1] §4.2 and [2] Appendix A:

      x_full[t] = 100 * ln(P_t / P_{t-n})        # n-bar momentum, length N - n
      y[t]      = x_full[t+1] - x_full[t]         # one-bar increment, length N - n - 1
      x[t]      = x_full[t]                       # drop the LAST element, not the first
      dy[t]     = y[t+1] - y[t]                   # acceleration, length N - n - 2

    Returned (x_t, y_t, dy_t) are all length N - n - 2, aligned so that the
    regression dy = -k*x_t - Omega*y_t fits the SDHO acceleration as a function
    of the position-velocity state at the START of the increment, not the end.

    Pairing y with the END position instead of the START position introduces
    a fencepost error that shifts the fitted Omega by approximately k.

    Parameters
    ----------
    prices : np.ndarray
        Closing prices (must be a 1-D array, no NaNs).
    n : int
        Look-back period in bars (default 5, matching [1] and [2]).

    Returns
    -------
    (x_t, y_t, dy) : tuple of np.ndarray, all length N - n - 2
    """
    log_prices = np.log(prices)
    x_full = 100.0 * (log_prices[n:] - log_prices[:-n])  # length N - n
    y = np.diff(x_full)                                   # length N - n - 1
    x = x_full[:-1]                                       # length N - n - 1
    dy = np.diff(y)                                       # length N - n - 2
    x_t = x[:-1]                                          # length N - n - 2
    y_t = y[:-1]                                          # length N - n - 2
    return x_t, y_t, dy


def fit_sdho(
    x: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
) -> tuple[float, float, float]:
    """OLS fit of dy = -k*x - Omega*y. Returns (k, Omega, R^2)."""
    X = np.column_stack([-x, -y])
    beta, *_ = np.linalg.lstsq(X, dy, rcond=None)
    k, Omega = float(beta[0]), float(beta[1])
    dy_pred = X @ beta
    ss_res = float(np.sum((dy - dy_pred) ** 2))
    ss_tot = float(np.sum((dy - dy.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot
    return k, Omega, r_squared


def r2_exact(k: float, Omega: float) -> float:
    """The exact discrete-time closed-form R^2 from Appendix A, equation (A.1).

    R^2_exact(k, Omega) = (Omega^2 - 1.5*k*Omega + 0.5*k^2 + k) / (2*Omega - k)
    """
    return (Omega ** 2 - 1.5 * k * Omega + 0.5 * k ** 2 + k) / (2 * Omega - k)


def r2_contact(Omega: float) -> float:
    """The contact-geometric leading-order form, equation (1) of the body.

    R^2 = Omega^2 / (1 + Omega^2)

    This is the first-order Taylor expansion of r2_exact(k, Omega) around
    the critical point (Omega = 1, k = 0).
    """
    return Omega ** 2 / (1 + Omega ** 2)
