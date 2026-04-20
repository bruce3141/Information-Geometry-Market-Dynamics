"""
p3_1.consistency_tests

Stationary-linear-SDHO consistency checks for each fitted market. At stationarity
with white noise, a 2D linear SDHO satisfies four identities simultaneously:

  1. Discrete Lyapunov equation: Sigma = A*Sigma*A^T + bb^T
  2. FDT ratio: sigma^2 / Var(y) = (4*Omega - 4*k + 3*k*Omega - 2*Omega^2 - k^2)/2
  3. Residual whiteness: sample autocorrelation at all lags -> 0
  4. Metric positive-definiteness: k, Omega, sigma > 0 (Fisher metric PD)

Deviations from any one of these carry direction-of-failure information:

  - Above-the-frontier deviations (R^2_measured > R^2_exact at same (k, Omega))
    show as drift-mismatch in the Lyapunov residual matrix (Sigma_yy or Cov(x,y)
    components off), and possibly as non-Gaussianity of residuals. Physically:
    more deterministic structure than the linear 2D SDHO permits (drift
    nonlinearity, under-resolved state, or broken stationarity).

  - Below-the-frontier deviations (R^2_measured < R^2_exact) show as nonzero
    residual autocorrelation and/or FDT-ratio departure from unity. Physically:
    more noise than the white-noise FDT stationarity allows (colored noise,
    regime mixing within the fit window).

The empirical claim of [Dean 2026] §3.2 that 23 markets sit on the exact
frontier within 4.5e-4 is equivalent to the simultaneous passing of this
four-way battery; reporting the battery in Table B3 makes that equivalence
explicit.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_lyapunov


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------

def sdho_matrices(k: float, Omega: float, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Discrete-time SDHO drift matrix A and noise covariance bb^T.

    State vector z_t = (x_t, y_t). Dynamics:
        x_{t+1} = x_t + y_t
        y_{t+1} = -k*x_t + (1-Omega)*y_t + sigma*epsilon_t
    so A is the 2x2 transition matrix and bb^T is rank-1 with the single
    nonzero entry sigma^2 in the (2,2) slot.
    """
    A = np.array([[1.0, 1.0],
                  [-k, 1.0 - Omega]])
    bbT = np.array([[0.0, 0.0],
                    [0.0, sigma ** 2]])
    return A, bbT


def empirical_sigma(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Empirical 2x2 state covariance for aligned (x, y) phase-space arrays."""
    x0 = x - x.mean()
    y0 = y - y.mean()
    c_xx = float(np.mean(x0 * x0))
    c_yy = float(np.mean(y0 * y0))
    c_xy = float(np.mean(x0 * y0))
    return np.array([[c_xx, c_xy], [c_xy, c_yy]])


def predicted_sigma(k: float, Omega: float, sigma: float) -> np.ndarray:
    """Stationary covariance from the discrete Lyapunov equation."""
    A, bbT = sdho_matrices(k, Omega, sigma)
    return solve_discrete_lyapunov(A, bbT)


# -----------------------------------------------------------------------------
# Individual tests
# -----------------------------------------------------------------------------

def lyapunov_residual(x: np.ndarray, y: np.ndarray,
                      k: float, Omega: float, sigma: float) -> dict:
    """Residual of the discrete Lyapunov equation at fitted parameters.

    At stationarity, Sigma_empirical should match Sigma_predicted component by
    component up to sampling error. The residual matrix is
        Delta = Sigma_emp - (A Sigma_emp A^T + bb^T)

    Sign conventions:
      Delta_xx > 0 : excess low-frequency energy (position variance)
      Delta_yy > 0 : excess velocity variance (below-curve signature)
      Delta_xy != 0: position-velocity coupling mismatch (drift nonlinearity
                     or phase-space dimensionality issue)
    """
    Sigma_emp = empirical_sigma(x, y)
    Sigma_pred = predicted_sigma(k, Omega, sigma)
    delta = Sigma_emp - Sigma_pred
    frob = float(np.linalg.norm(delta, ord='fro'))
    frob_emp = float(np.linalg.norm(Sigma_emp, ord='fro'))
    return {
        'frobenius': frob,
        'relative': frob / frob_emp if frob_emp > 0 else float('nan'),
        'delta_xx': float(delta[0, 0]),
        'delta_yy': float(delta[1, 1]),
        'delta_xy': float(delta[0, 1]),
        'sigma_emp_yy': float(Sigma_emp[1, 1]),
        'sigma_pred_yy': float(Sigma_pred[1, 1]),
    }


def residual_autocorr(residuals: np.ndarray, max_lag: int = 10) -> dict:
    """Sample autocorrelations of SINDy residuals at lags 1..max_lag.

    Under the white-noise stationarity assumption, all autocorrelations vanish
    in the large-sample limit. Significant positive lag-1 is the signature of
    colored noise, pushing the system below the frontier.
    """
    r = np.asarray(residuals, dtype=float)
    r = r - r.mean()
    denom = float(np.sum(r * r))
    if denom == 0:
        return {'lag1': float('nan'), 'lag5': float('nan'), 'max_abs': float('nan')}
    acfs = np.array([
        float(np.sum(r[:-k] * r[k:]) / denom) for k in range(1, max_lag + 1)
    ])
    return {
        'lag1': float(acfs[0]),
        'lag5': float(acfs[4]) if max_lag >= 5 else float('nan'),
        'max_abs': float(np.max(np.abs(acfs))),
        'max_lag_reported': int(max_lag),
    }


def fdt_ratio(y: np.ndarray, k: float, Omega: float, sigma: float) -> dict:
    """Fluctuation-dissipation ratio: measured vs theoretical.

    From Appendix A, the stationary Lyapunov solution gives
        sigma^2 / Var(y) = (4*Omega - 4*k + 3*k*Omega - 2*Omega^2 - k^2) / 2
    so the ratio measured / theoretical should equal 1 at stationarity.
    """
    var_y = float(np.var(y, ddof=0))
    theoretical = (4 * Omega - 4 * k + 3 * k * Omega - 2 * Omega ** 2 - k ** 2) / 2
    measured = (sigma ** 2) / var_y if var_y > 0 else float('nan')
    ratio = measured / theoretical if theoretical not in (0, 0.0) else float('nan')
    return {
        'measured': measured,
        'theoretical': theoretical,
        'ratio': ratio,
        'deviation_pct': 100 * (ratio - 1.0) if np.isfinite(ratio) else float('nan'),
    }


def metric_pd(k: float, Omega: float, sigma: float) -> dict:
    """Fisher metric positive-definiteness check.

    The Fisher metric on (k, Omega, sigma), computed in [1] §6.2, has diagonal
    entries 1/(2*k*Omega), 1/(2*Omega), 2/sigma^2. All three are positive only
    when k, Omega, sigma > 0. If any parameter turns negative under the fit,
    Cencov's uniqueness theorem is inapplicable and the geodesic distance
    between fitted parameter points acquires an imaginary component.
    """
    return {
        'k_positive': bool(k > 0),
        'Omega_positive': bool(Omega > 0),
        'sigma_positive': bool(sigma > 0),
        'all_positive': bool((k > 0) and (Omega > 0) and (sigma > 0)),
    }


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------

def run_all(x: np.ndarray, y: np.ndarray, dy: np.ndarray,
            k: float, Omega: float,
            sigma: float | None = None) -> dict:
    """Run the full four-test consistency battery on a fitted market.

    Parameters
    ----------
    x, y, dy : phase-space arrays from build_phase_space()
    k, Omega : fitted SDHO drift parameters from fit_sdho()
    sigma    : noise scale; if None, computed as population-std of the
               regression residual dy - (-k*x - Omega*y)
    """
    residuals = np.asarray(dy) - (-k * np.asarray(x) - Omega * np.asarray(y))
    if sigma is None:
        sigma = float(np.std(residuals, ddof=0))
    return {
        'params': {'k': float(k), 'Omega': float(Omega), 'sigma': float(sigma)},
        'lyapunov': lyapunov_residual(x, y, k, Omega, sigma),
        'autocorr': residual_autocorr(residuals, max_lag=10),
        'fdt': fdt_ratio(y, k, Omega, sigma),
        'metric_pd': metric_pd(k, Omega, sigma),
    }


def summary_row(result: dict) -> dict:
    """Flatten a run_all result into a single-row dict for tabular output.

    Columns chosen for Table B3: relative Lyapunov residual (dimensionless
    Frobenius norm), lag-1 residual autocorrelation, FDT ratio, and a Boolean
    metric-PD flag.
    """
    return {
        'k':              result['params']['k'],
        'Omega':          result['params']['Omega'],
        'sigma':          result['params']['sigma'],
        'lyap_relative':  result['lyapunov']['relative'],
        'lag1_acf':       result['autocorr']['lag1'],
        'fdt_ratio':      result['fdt']['ratio'],
        'metric_pd_pass': result['metric_pd']['all_positive'],
    }


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

def main():
    """Run the four-way consistency battery on the Appendix B panel.

    Downloads daily prices via yfinance, fits the SDHO at n=5, and reports the
    four consistency-test statistics per market, producing the data behind
    Table B3 of the paper.
    """
    import numpy as np
    import yfinance as yf

    from p3_1.phase_space import build_phase_space, fit_sdho, r2_exact
    from p3_1.falsification_test import MARKETS as APPENDIX_B_MARKETS

    header = (
        f"{'Market':<7} | {'n_obs':>5} | {'k':>6} | {'Omega':>6} | "
        f"{'R^2':>6} | {'Rexact':>7} | {'resid':>9} | "
        f"{'lyap_rel':>8} | {'lag1':>7} | {'fdt_r':>6} | {'PD':>3}"
    )
    print(header)
    print('-' * len(header))

    rows = []
    for ticker, label, _desc, start in APPENDIX_B_MARKETS:
        df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
        if df.empty or len(df) < 2000:
            print(f"{label:<7} | ERROR: insufficient data")
            continue
        prices = np.asarray(df['Close']).ravel()
        x, y, dy = build_phase_space(prices, n=5)
        k, Omega, R2 = fit_sdho(x, y, dy)
        result = run_all(x, y, dy, k, Omega)
        row = summary_row(result)
        row['label']       = label
        row['n_obs']       = len(x)
        row['R2_data']     = R2
        row['R2_exact']    = r2_exact(k, Omega)
        row['R2_residual'] = R2 - row['R2_exact']
        rows.append(row)
        print(
            f"{row['label']:<7} | {row['n_obs']:>5d} | {row['k']:>6.3f} | "
            f"{row['Omega']:>6.3f} | {row['R2_data']:>6.3f} | "
            f"{row['R2_exact']:>7.4f} | {row['R2_residual']:>+9.1e} | "
            f"{row['lyap_relative']:>8.2e} | {row['lag1_acf']:>+7.4f} | "
            f"{row['fdt_ratio']:>6.4f} | {str(row['metric_pd_pass']):>3}"
        )

    if rows:
        lyap = np.array([r['lyap_relative'] for r in rows])
        lag1 = np.array([r['lag1_acf']       for r in rows])
        fdt  = np.array([r['fdt_ratio']      for r in rows])
        pd_all = all(r['metric_pd_pass']     for r in rows)
        print()
        print(f"Summary across {len(rows)} markets:")
        print(f"  lyap_relative : max = {lyap.max():.2e}, median = {np.median(lyap):.2e}")
        print(f"  lag1_acf      : min = {lag1.min():+.4f}, max = {lag1.max():+.4f}, mean = {lag1.mean():+.4f}")
        print(f"  fdt_ratio     : min = {fdt.min():.4f}, max = {fdt.max():.4f}")
        print(f"  metric_pd     : {pd_all} for all {len(rows)} markets")


if __name__ == "__main__":
    main()
