#!/bin/bash
# Render the full sim_v1 dataset (6,400 scenes, ~26 GB) exactly as used for the published models.
#   bash scripts/render_dataset.sh [workers]
# ~4 s CPU per scene: about 1 h with 8 workers. Restartable: finished scenes are skipped.
# Seed ranges (split = seed % 20: 0 -> val, 1 -> test_sim, else train; held-out scenes use seed + 10,000,000):
#   p0000000-p0003999   seeds 0-3999                         train/val/test_sim
#   a1000000-a1001999   seeds 1,000,000-1,001,999            train/val/test_sim
#   ah0000000-ah0000399 seeds 0-399 with --heldout           test_heldout
set -euo pipefail
cd "$(dirname "$0")/.."
W=${1:-8}
PY=.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
$PY -m sim.render --out data/sim_v1 --start 0       --count 4000 --workers "$W" --prefix p
$PY -m sim.render --out data/sim_v1 --start 1000000 --count 2000 --workers "$W" --prefix a
$PY -m sim.render --out data/sim_v1 --start 0       --count 400  --workers "$W" --prefix a --heldout
$PY scripts/manifest.py verify data/sim_v1 results/sim_v1_manifest.csv
