#!/bin/bash
# Render only the two test splits (700 scenes, ~3 GB): enough to evaluate the published models.
#   bash scripts/render_test.sh [workers]
set -euo pipefail
cd "$(dirname "$0")/.."
W=${1:-8}
PY=.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
$PY -m sim.render --out data/sim_v1 --start 0       --count 4000 --workers "$W" --prefix p --only-split test_sim
$PY -m sim.render --out data/sim_v1 --start 1000000 --count 2000 --workers "$W" --prefix a --only-split test_sim
$PY -m sim.render --out data/sim_v1 --start 0       --count 400  --workers "$W" --prefix a --heldout
$PY scripts/manifest.py verify data/sim_v1 results/sim_v1_manifest.csv --split test_sim test_heldout
