#!/bin/bash
# Render the localisation scenes of one configuration (tasks/phase1_localisation.md step 1), then run
# direction finding on them (step 2): all variants on the first 100 scenes, the main methods on the rest.
#   bash scripts/render_loc.sh <config> [workers] [count]
# Configs: tetra_pole_100, flat_ground_100, tetra_ground_100, tetra_pole_150 (sim/layouts.py CONFIGS).
# ~24 MB and ~175 CPU-s per scene; ~1.6 GB RAM per worker.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=$1
W=${2:-8}
N=${3:-300}
PY=.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
$PY -m loc.render --config "$CFG" --count "$N" --workers "$W"
$PY -m loc.run_doa --config "$CFG" --variants --limit 100 --wait-for 100
$PY -m loc.run_doa --config "$CFG" --wait-for "$N"
echo "DONE render_loc $CFG"
