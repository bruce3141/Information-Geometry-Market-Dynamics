#!/usr/bin/env bash
# scripts/build_pdf.sh
#
# Build an Entropy-style PDF from the current draft using pandoc + XeLaTeX.
#
# Requirements:
#   - pandoc >= 3.0
#   - XeLaTeX (via TeX Live, MacTeX, or MiKTeX)
#   - TeX Gyre Pagella font (texlive-fonts-recommended on Debian/Ubuntu)
#
# Run from repository root:
#   bash scripts/build_pdf.sh
#
# Output: paper/p3_1_v57.pdf

set -e

# Locate repo root relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Sanity checks
command -v pandoc >/dev/null 2>&1 || {
    echo "ERROR: pandoc not found. Install pandoc >= 3.0."
    exit 1
}
command -v xelatex >/dev/null 2>&1 || {
    echo "ERROR: xelatex not found. Install TeX Live or equivalent."
    exit 1
}

python3 "$REPO_ROOT/scripts/build_pdf.py"
