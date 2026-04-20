#!/usr/bin/env bash
# scripts/reproduce_all.sh
#
# One-command reproduction of all P3-1 results that are not directly imported
# from companion papers [1] and [2].
#
# Run from the repository root:
#   bash scripts/reproduce_all.sh
#
# Total runtime ~ 2-3 minutes, dominated by Appendix B yfinance downloads.

set -e  # fail-fast

echo "================================================================="
echo "P3-1 Full Reproduction"
echo "Dean (2026), 'Information Geometry of Market Dynamics'"
echo "================================================================="

echo ""
echo "[1/7] Generating paper figures 1 and 2..."
p3_1_figures

echo ""
echo "[2/7] Generating Figure 3 (phase-plane flow alignment, requires internet)..."
p3_1_figure3 --both

echo ""
echo "[3/7] Verifying Appendix C Pareto frontier theorem (4 numerical tests)..."
p3_1_pareto_verify

echo ""
echo "[4/7] Running §4.2-footnote Sharpe-optimization demonstration..."
p3_1_sharpe

echo ""
echo "[5/7] Running Appendix B 17-market falsification test (~ 1 minute)..."
echo "      Note: requires internet access for yfinance downloads."
p3_1_falsification

echo ""
echo "[6/7] Running §B.5 four-way stationary-SDHO consistency battery..."
echo "      Note: requires internet access for yfinance downloads."
p3_1_consistency

echo ""
echo "[7/7] Running §6.2 underdamped-regime candidate sweep (~ 30 sec)..."
echo "      Note: requires internet access for yfinance downloads."
p3_1_underdamped

echo ""
echo "================================================================="
echo "Reproduction complete."
echo "Figures saved to figures/."
echo "================================================================="
