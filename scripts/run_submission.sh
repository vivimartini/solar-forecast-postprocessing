#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=.

python -m pytest tests/ -q
python scripts/00_scope_analysis.py
python scripts/01_eda.py
python scripts/02_baseline.py
python scripts/10_final_test.py
python scripts/16_significance.py
python scripts/make_figures.py

echo ""
echo "Done. Key outputs:"
echo "  outputs/eda_diagnostics.png"
echo "  outputs/predictions.csv"
echo "  outputs/fig_*.png"
