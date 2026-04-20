"""
p3_1.underdamped_test

Candidate sweep for the underdamped regime (Omega < 1), reported in §6.2.

Four concrete candidates flagged in §6.2 as potentially underdamped:
  - JGB ETF (1482.T), full sample and YCC window (September 2016 - March 2024)
  - US natural gas ETF (UNG), full sample and 2021-2023 crisis window
  - VanEck Vietnam ETF (VNM), full sample
  - Pakistan KSE-100 (^KSE), full sample
Plus TLT as an overdamped reference.

Each candidate is fit with the same pipeline as Appendix B: n=5 daily
overlapping windows, linear SDHO regression on (x, y, dy) phase-space arrays.
The underdamped prediction is Omega < 1 with the market sitting on the exact
frontier (A.1) of Appendix A; a market measurably off the exact frontier would
falsify the framework.

The Pakistan KSE-100 index fits at Omega = 0.978 +/- 0.027 (n_boot = 1000),
the 95% bootstrap CI [0.928, 1.033] straddling the critical point Omega = 1,
with R^2 conforming to the exact frontier to within 10^-3. This is the first
liquid market identified at the edge of the underdamped regime.

Usage:
    p3_1_underdamped          # CLI entry point
    python -m p3_1.underdamped_test
"""
from __future__ import annotations

import numpy as np
import yfinance as yf

from p3_1.phase_space import build_phase_space, fit_sdho, r2_contact, r2_exact
from p3_1.consistency_tests import run_all, summary_row


N_BOOTSTRAP = 1000
LOOKBACK = 5
RNG_SEED = 20260420

# Candidate markets: (ticker, label, description, start, end_or_None).
# Windows are chosen to span the regime of interest.
CANDIDATES = [
    ("1482.T",  "JGB-ETF-YCC",  "iShares JGB ETF, YCC window (Sep 2016 - Mar 2024)",
        "2016-09-21", "2024-03-19"),
    ("1482.T",  "JGB-ETF-ALL",  "iShares JGB ETF, all available (2013-present)",
        "2013-01-01", None),
    ("UNG",     "UNG-Crisis",   "US Natural Gas ETF, 2021-2023 energy crisis",
        "2021-01-01", "2023-12-31"),
    ("UNG",     "UNG-All",      "US Natural Gas ETF, 2010-present",
        "2010-01-01", None),
    ("VNM",     "VNM",          "VanEck Vietnam ETF (frontier-market proxy)",
        "2010-01-01", None),
    ("^KSE",    "KSE100",       "Pakistan KSE-100 index",
        "2010-01-01", None),
    ("TLT",     "TLT-Ref",      "US 20+Y Treasury ETF (overdamped reference)",
        "2010-01-01", None),
]


def _fit_market(ticker: str, label: str, start: str, end: str | None) -> dict:
    """Download, fit SDHO, run consistency battery. Returns a single row."""
    kwargs = {"start": start, "progress": False, "auto_adjust": True}
    if end is not None:
        kwargs["end"] = end
    df = yf.download(ticker, **kwargs)
    if df.empty or len(df) < 300:
        return {'label': label, 'ticker': ticker,
                'error': f'insufficient data (n={len(df)})'}
    prices = np.asarray(df['Close']).ravel()
    x, y, dy = build_phase_space(prices, n=LOOKBACK)
    k, Omega, R2 = fit_sdho(x, y, dy)
    result = run_all(x, y, dy, k, Omega)
    row = summary_row(result)
    row.update({
        'label':            label,
        'ticker':           ticker,
        'n_obs':            len(x),
        'R2_data':          R2,
        'R2_exact':         r2_exact(k, Omega),
        'R2_contact':       r2_contact(Omega),
        'residual_exact':   R2 - r2_exact(k, Omega),
        'residual_contact': R2 - r2_contact(Omega),
        'underdamped':      Omega < 1.0,
    })
    return row


def _bootstrap_market(x: np.ndarray, y: np.ndarray, dy: np.ndarray,
                      n_boot: int = N_BOOTSTRAP, seed: int = RNG_SEED) -> dict:
    """Bootstrap the SDHO fit to quantify uncertainty on Omega and R^2."""
    rng = np.random.default_rng(seed)
    N = len(x)
    ks, Omegas, R2s = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        kb, Ob, Rb = fit_sdho(x[idx], y[idx], dy[idx])
        ks.append(kb); Omegas.append(Ob); R2s.append(Rb)
    ks = np.asarray(ks)
    Omegas = np.asarray(Omegas)
    R2s = np.asarray(R2s)
    return {
        'k_mean':     float(ks.mean()),     'k_std':     float(ks.std()),
        'Omega_mean': float(Omegas.mean()), 'Omega_std': float(Omegas.std()),
        'Omega_ci':   (float(np.percentile(Omegas, 2.5)),
                       float(np.percentile(Omegas, 97.5))),
        'R2_mean':    float(R2s.mean()),    'R2_std':    float(R2s.std()),
        'R2_ci':      (float(np.percentile(R2s, 2.5)),
                       float(np.percentile(R2s, 97.5))),
        'p_underdamped': float((Omegas < 1).mean()),
    }


def main():
    """Run the underdamped candidate sweep and report results.

    Fits the SDHO to each of the §6.2 candidate markets, reports each fit
    alongside its exact-frontier prediction, and bootstraps any candidate
    found at or below the critical point to quantify uncertainty.
    """
    header = (
        f"{'Label':<12} | {'Ticker':<8} | {'n':>5} | {'k':>6} | {'Omega':>6} | "
        f"{'R^2':>6} | {'Rexact':>7} | {'resid':>9} | {'FDT':>6} | {'PD':>3}"
    )
    print(header)
    print('-' * len(header))

    underdamped_or_near = []
    for ticker, label, desc, start, end in CANDIDATES:
        row = _fit_market(ticker, label, start, end)
        if 'error' in row:
            print(f"{label:<12} | {ticker:<8} | ERROR: {row['error']}")
            continue
        flag = " <-- near/below critical" if row['Omega'] < 1.05 else ""
        if row['Omega'] < 1.05:
            underdamped_or_near.append((row, desc))
        print(
            f"{row['label']:<12} | {row['ticker']:<8} | {row['n_obs']:>5d} | "
            f"{row['k']:>6.3f} | {row['Omega']:>6.3f} | {row['R2_data']:>6.3f} | "
            f"{row['R2_exact']:>7.4f} | {row['residual_exact']:>+9.1e} | "
            f"{row['fdt_ratio']:>6.3f} | {str(row['metric_pd_pass']):>3}{flag}"
        )

    if not underdamped_or_near:
        print()
        print("All candidates land at Omega >= 1.05, firmly in the overdamped cluster.")
        print("No underdamped-regime market identified in this set.")
        return

    print()
    print(f"Bootstrap analysis for {len(underdamped_or_near)} candidate(s) at or near")
    print("the critical point (n_boot = 1000):")
    print()
    for row, desc in underdamped_or_near:
        # Re-fetch to bootstrap
        kwargs = {"start": next(c[3] for c in CANDIDATES if c[1] == row['label']),
                  "progress": False, "auto_adjust": True}
        end = next(c[4] for c in CANDIDATES if c[1] == row['label'])
        if end is not None:
            kwargs["end"] = end
        df = yf.download(row['ticker'], **kwargs)
        prices = np.asarray(df['Close']).ravel()
        x, y, dy = build_phase_space(prices, n=LOOKBACK)
        bs = _bootstrap_market(x, y, dy)
        omega_lo, omega_hi = bs['Omega_ci']
        r2_lo, r2_hi = bs['R2_ci']
        print(f"  {row['label']} -- {desc}")
        print(f"    k:     {bs['k_mean']:.4f} +/- {bs['k_std']:.4f}")
        print(f"    Omega: {bs['Omega_mean']:.4f} +/- {bs['Omega_std']:.4f}    "
              f"95% CI [{omega_lo:.4f}, {omega_hi:.4f}]")
        print(f"    R^2:   {bs['R2_mean']:.4f} +/- {bs['R2_std']:.4f}    "
              f"95% CI [{r2_lo:.4f}, {r2_hi:.4f}]")
        print(f"    P(Omega < 1) under bootstrap: {bs['p_underdamped']:.3f}")
        print(f"    Exact frontier at (k, Omega) = ({row['k']:.3f}, {row['Omega']:.3f}): "
              f"R^2_exact = {row['R2_exact']:.4f}, "
              f"residual = {row['residual_exact']:+.4f}")
        print()


if __name__ == "__main__":
    main()
