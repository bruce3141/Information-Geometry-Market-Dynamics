# Information Geometry of Market Dynamics

Reproducibility code for:

> Dean, B.H. (2026). *Information Geometry of Market Dynamics: A Pareto Frontier from Contact Geometry.*

Working draft and pre-built PDF: `paper/p3_1_v60.md` and `paper/p3_1_v60.pdf`.

This repository regenerates all results in the paper that are not directly imported from companion papers [1] and [2]: the seventeen-market cross-market validation (Appendix B, Tables B1 and B2), the four-way stationary-SDHO consistency battery (Appendix B §B.5, Table B3), the phase-plane flow alignment for SPY (Appendix B §B.6, Figure 3), the §6.2 underdamped-regime candidate sweep (including the KSE-100 critical-point result), the four numerical verifications of the Pareto frontier theorem (Appendix C), the §4.2-footnote Sharpe-optimization demonstration, and all paper figures.

The six futures markets in Table 1 are imported from [1], Table 2; the underlying Databento price data are not redistributed.

## Install

Requires Python 3.10 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

(`uv venv` and conda also work; substitute the activation step accordingly.)

## Reproduce all results

One command, end-to-end, about 3-4 minutes:

```bash
bash scripts/reproduce_all.sh
```

## Individual stages

```bash
p3_1_figures           # Paper Figures 1 and 2 (both themes)
p3_1_figure3           # Paper Figure 3, phase-plane flow alignment (requires internet)
p3_1_pareto_verify     # Appendix C, 4 numerical Pareto-frontier tests
p3_1_sharpe            # §4.2 footnote, Sharpe-optimization demonstration
p3_1_falsification     # Appendix B, 17-market validation (~1 min, requires internet)
p3_1_consistency       # §B.5, four-way Lyapunov-consistency battery (~1 min, requires internet)
p3_1_underdamped       # §6.2 underdamped candidate sweep (~30 sec, requires internet)
```

The `p3_1_falsification` test downloads daily price data from Yahoo Finance via `yfinance` and includes a sanity check against [2], Table 7a (SPY 1993-2025) reproducing the published (k, Omega, R^2) to four decimal places. If the sanity check fails, the fencepost-bug regression in `phase_space.py` has been broken. The `p3_1_consistency` script uses the same download path and additionally reports, per market, the Frobenius norm of the Lyapunov residual relative to the empirical covariance, the lag-1 autocorrelation of the SINDy residual, the FDT ratio against its theoretical value from Appendix A, and a Boolean metric positive-definiteness flag. Across the 17-market panel, all four tests pass within sampling bounds.

The `p3_1_underdamped` sweep tests the §6.2 candidate markets (JGB ETF, US natural gas, Vietnam ETF, Pakistan KSE-100, TLT reference) for Ω < 1. The Pakistan KSE-100 index fits at Ω = 0.978 ± 0.027, 95% CI [0.928, 1.033], the first liquid market identified at or near the critical point Ω = 1.

## Figures

**Figure 1** shows the Pareto frontier R² = Ω² / (1 + Ω²) with all 23 Appendix-B markets marked. The region above the curve is labeled DETERMINISTIC DOMINATED (red), below is STOCHASTIC DOMINATED (green). The detailed at-market-k interpretation of each region is in §C.4; the empirical test of the SDHO identity is in §B.5 and Figure 3.

**Figure 2** shows the same detail window with the exact discrete-time closed form R²_exact(k, Ω) from Appendix A, tightening the 23-market residuals from ~10⁻² against the contact-geometric form to ~10⁻⁴.

**Figure 3** shows phase-plane flow alignment for SPY: the fitted SDHO drift streamplot in panel (a), and the empirical trajectory density with the fitted drift streamplot and Lyapunov-predicted covariance ellipses overlaid in panel (b). The alignment between predicted ellipses and empirical density, without any fitted angle or scale, is the phase-plane version of the §B.5 four-way battery.

Both dark-theme and light-theme versions of each figure are generated. Light-theme versions live in both `figures/` and `paper/` (the `paper/` copies are embedded in the pre-built PDF). Dark-theme versions live only in `figures/`.

## Code organization

```
src/p3_1/
  phase_space.py                 Shared phase-space construction, SDHO fit, R^2 closed forms
  falsification_test.py          Appendix B, cross-market validation (Tables B1, B2)
  consistency_tests.py           Appendix B §B.5, stationary-SDHO consistency battery (Table B3)
  figure3.py                     Appendix B §B.6, phase-plane flow alignment (Figure 3)
  underdamped_test.py            §6.2 underdamped-regime candidate sweep
  pareto_bound_verification.py   Appendix C, Pareto frontier numerical tests
  sharpe_optimization.py         Section 4.2 footnote, economic-attractor demonstration
  figures.py                     Paper Figures 1 and 2, both themes
paper/
  p3_1_v60.md                    Working draft (Markdown)
  p3_1_v60.pdf                   Pre-built PDF
  Figure*.png                    Light-theme figures (embedded in the PDF)
figures/                         Dark-theme + light-theme duplicates
scripts/
  reproduce_all.sh               One-command reproduction (7 stages)
```

The consistency tests module implements four independent probes of the stationary-linear-SDHO-with-white-noise assumption underlying the frontier identity: (1) the discrete Lyapunov equation residual `Sigma_empirical - (A*Sigma_empirical*A^T + bb^T)`, reported as Frobenius norm relative to `|Sigma_empirical|_F`; (2) the lag-1 sample autocorrelation of the SINDy residuals; (3) the FDT ratio `sigma^2 / Var(y)` compared to the theoretical value `(4*Omega - 4*k + 3*k*Omega - 2*Omega^2 - k^2)/2` from the Appendix A Lyapunov solution; and (4) the sign constraints `k, Omega, sigma > 0` required for Fisher-metric positive-definiteness and applicability of Cencov's uniqueness theorem. Deviations from any one test carry direction-of-failure information: above-frontier deviations manifest as drift-mismatch signatures in the Lyapunov residual; below-frontier deviations manifest as nonzero residual autocorrelation or FDT-ratio departure from unity.

## License

MIT. See `LICENSE`.

## Citation

```bibtex
@article{Dean2026P3-1,
  author  = {Dean, Bruce H.},
  title   = {Information Geometry of Market Dynamics: A Pareto Frontier from Contact Geometry},
  journal = {Working paper},
  year    = {2026}
}
```

## References

[1] Dean, B.H. *Scale Invariant Dynamics in Market Price Momentum.* SSRN Working Paper, 2025. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5990674

[2] Dean, B.H. *Scale-Dependent Dynamics in Equity Market Phase Space.* SSRN Working Paper, 2026. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6380118
