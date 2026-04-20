"""
p3_1.falsification_test

Cross-market falsification of the Pareto frontier formula on seventeen markets
beyond the original six futures of [1]. Reproduces Tables B1 and B2 of
Appendix B of Dean (2026).

Methodology (matching [1] and [2] exactly):
  - Daily close data from Yahoo Finance via yfinance
  - Phase-space coordinates: x_t = 100 * ln(P_t / P_{t-n}) with n = 5
                             y_t = x_full[t+1] - x_full[t]
  - OLS fit of dy = -k*x - Omega*y on the phase-space transitions
  - R^2 computed on the dy regression
  - Bootstrap 95% CI for Omega and R^2 over n_bootstrap resamples
  - Frontier prediction: R^2_frontier = Omega^2 / (1 + Omega^2)
  - Residual: R^2_data - R^2_frontier

The script also implements a sanity check against [2], Table 7a (SPY
1993-2025 at n=5 daily, overlapping, linear), expected to match the published
(k, Omega, R^2) to four decimal places. If the sanity check fails, the
fencepost-bug regression test in src/p3_1/phase_space.py has likely been broken.

Usage:
    p3_1_falsification              # CLI entry point
    python -m p3_1.falsification_test
"""
from __future__ import annotations

import numpy as np
import yfinance as yf

from p3_1.phase_space import build_phase_space, fit_sdho, r2_contact, r2_exact

RNG = np.random.default_rng(20260413)
LOOKBACK = 5
N_BOOTSTRAP = 1000
START = "2010-01-01"

SANITY_CHECK = ("SPY", "SPY", "Pipeline sanity check against P2 Table 7a", "1993-01-01")

MARKETS = [
    # --- Asian developed-market equities ---
    ("^HSI",       "HSI",    "Hang Seng Index (Hong Kong)",      START),
    ("^AXJO",      "AXJO",   "ASX 200 (Australia)",              START),
    ("^N225",      "N225",   "Nikkei 225 (Japan)",               START),
    ("^KS11",      "KOSPI",  "KOSPI (South Korea)",              START),
    ("^TWII",      "TSEC",   "TSEC Weighted (Taiwan)",           START),
    # --- European equities ---
    ("^FTSE",      "FTSE",   "FTSE 100 (UK)",                    START),
    ("^GDAXI",     "DAX",    "DAX (Germany)",                    START),
    ("^FCHI",      "CAC",    "CAC 40 (France)",                  START),
    ("^STOXX50E",  "STOXX",  "Euro Stoxx 50",                    START),
    # --- Emerging-market equities ---
    ("^BVSP",      "BVSP",   "Bovespa (Brazil)",                 START),
    ("^NSEI",      "NIFTY",  "NIFTY 50 (India)",                 START),
    ("^MXX",       "IPC",    "IPC (Mexico)",                     START),
    ("^JKSE",      "JKSE",   "Jakarta Composite (Indonesia)",    START),
    # --- Cryptocurrencies ---
    ("BTC-USD",    "BTC",    "Bitcoin",                          "2014-09-17"),
    ("ETH-USD",    "ETH",    "Ethereum",                         "2017-11-09"),
    ("BNB-USD",    "BNB",    "Binance Coin",                     "2017-11-09"),
    ("XRP-USD",    "XRP",    "XRP",                              "2017-11-09"),
]


def bootstrap_ci(x, y, dy, n_boot=N_BOOTSTRAP):
    n = len(x)
    ks, omegas, r2s, resids = [], [], [], []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        k_b, Omega_b, r2_b = fit_sdho(x[idx], y[idx], dy[idx])
        if Omega_b <= 0:
            continue
        r2_front = r2_contact(Omega_b)
        ks.append(k_b)
        omegas.append(Omega_b)
        r2s.append(r2_b)
        resids.append(r2_b - r2_front)
    return {
        "k":        (float(np.percentile(ks, 2.5)),     float(np.percentile(ks, 97.5))),
        "Omega":    (float(np.percentile(omegas, 2.5)), float(np.percentile(omegas, 97.5))),
        "R2":       (float(np.percentile(r2s, 2.5)),    float(np.percentile(r2s, 97.5))),
        "residual": (float(np.percentile(resids, 2.5)), float(np.percentile(resids, 97.5))),
    }


def run_market(ticker, name, description, start):
    data = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    if data is None or len(data) == 0:
        return {"ticker": ticker, "error": "no data"}
    prices = data["Close"].to_numpy().flatten()
    prices = prices[~np.isnan(prices)]
    n_bars = len(prices)
    x, y, dy = build_phase_space(prices, n=LOOKBACK)
    k, Omega, r_squared = fit_sdho(x, y, dy)
    r2_frontier = r2_contact(Omega)            # leading-order, eq. (1)
    r2_frontier_exact = r2_exact(k, Omega)      # exact, eq. (A.1)
    residual = r_squared - r2_frontier
    residual_exact = r_squared - r2_frontier_exact
    cis = bootstrap_ci(x, y, dy)
    return {
        "ticker": ticker, "name": name, "description": description,
        "n_bars": n_bars, "n_fit": len(x),
        "k": k, "Omega": Omega,
        "R2_data": r_squared,
        "R2_frontier": r2_frontier,
        "R2_frontier_exact": r2_frontier_exact,
        "residual": residual,
        "residual_exact": residual_exact,
        "ci": cis,
    }


def print_result(r):
    if "error" in r:
        print(f"\n{r['ticker']}: ERROR ({r['error']})")
        return
    print(f"\n{r['name']} ({r['ticker']}) -- {r['description']}")
    print(f"  Sample:        {r['n_bars']} daily bars, {r['n_fit']} phase-space points")
    print(f"  k:             {r['k']:+.4f}   CI [{r['ci']['k'][0]:+.4f}, {r['ci']['k'][1]:+.4f}]")
    print(f"  Omega:         {r['Omega']:+.4f}   CI [{r['ci']['Omega'][0]:+.4f}, {r['ci']['Omega'][1]:+.4f}]")
    print(f"  R^2 (data):    {r['R2_data']:.4f}   CI [{r['ci']['R2'][0]:.4f}, {r['ci']['R2'][1]:.4f}]")
    print(f"  R^2 frontier:  {r['R2_frontier']:.4f}   (leading-order, eq. (1))")
    print(f"  R^2 exact:     {r['R2_frontier_exact']:.4f}   (exact, eq. (A.1))")
    print(f"  residual vs (3):    {r['residual']:+.4f}   CI [{r['ci']['residual'][0]:+.4f}, {r['ci']['residual'][1]:+.4f}]")
    print(f"  residual vs (A.1):  {r['residual_exact']:+.6f}")


def main():
    print("=" * 72)
    print("P3-1 Falsification Test: Cross-Market Validation of R^2 = Omega^2/(1+Omega^2)")
    print("=" * 72)
    print(f"Look-back: {LOOKBACK} bars (daily)")
    print(f"Bootstrap resamples: {N_BOOTSTRAP}")
    print(f"Dean (2025) reference band: Omega in [1.12, 1.21], R^2 in [0.56, 0.60]")

    print("\n" + "-" * 72)
    print("PIPELINE SANITY CHECK (SPY 1993-2025 daily, n=5 overlapping)")
    print("-" * 72)
    print("Expected from P2 Table 7a: k=0.260, Omega=1.184, R^2=0.585")
    sanity = run_market(*SANITY_CHECK)
    print_result(sanity)
    if "error" not in sanity:
        if abs(sanity["Omega"] - 1.184) > 0.005:
            print("\n*** WARNING *** SPY Omega off by >0.005. Fencepost bug regression?")

    print("\n" + "-" * 72)
    print("FALSIFICATION TEST MARKETS")
    print("-" * 72)
    results = []
    for ticker, name, desc, start in MARKETS:
        r = run_market(ticker, name, desc, start)
        results.append(r)
        print_result(r)

    print("\n" + "=" * 86)
    print("SUMMARY TABLE (falsification test markets)")
    print("=" * 86)
    print(f"{'Market':<8} {'n':>6} {'k':>8} {'Omega':>8} {'R2_data':>9} "
          f"{'R2_front':>9} {'res_(3)':>10} {'res_(A.1)':>13}")
    print("-" * 86)
    for r in results:
        if "error" in r:
            continue
        print(f"{r['name']:<8} {r['n_fit']:>6} {r['k']:>+8.4f} {r['Omega']:>+8.4f} "
              f"{r['R2_data']:>9.4f} {r['R2_frontier']:>9.4f} "
              f"{r['residual']:>+10.4f} {r['residual_exact']:>+13.6f}")


if __name__ == "__main__":
    main()
