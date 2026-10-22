#!/bin/bash
# One-time setup: Python venv, packages, the PANNs CNN14 checkpoint (only needed for training) and the trained models.
#   bash scripts/setup.sh [--cpu] [--no-train-weights]
# Run from the repo root. Needs Python 3.10, and the GitHub CLI (gh, logged in) for the trained models.
set -euo pipefail
cd "$(dirname "$0")/.."
CPU=0; TRAIN_WEIGHTS=1
for arg in "$@"; do
  case $arg in --cpu) CPU=1 ;; --no-train-weights) TRAIN_WEIGHTS=0 ;; esac
done

if [ ! -d .venv ]; then python3.10 -m venv .venv || python3 -m venv .venv; fi
PY=.venv/bin/python
$PY -m pip install -q --upgrade pip
if [ $CPU = 1 ]; then
  $PY -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
else
  $PY -m pip install -q torch --index-url https://download.pytorch.org/whl/cu128
fi
$PY -m pip install -q -r requirements.txt
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

mkdir -p weights models
if [ $TRAIN_WEIGHTS = 1 ] && [ ! -f "weights/Cnn14_DecisionLevelMax_mAP=0.385.pth" ]; then
  # PANNs (Kong et al. 2020), Zenodo record 3987831; 327,428,481 bytes
  curl -L --fail -o "weights/Cnn14_DecisionLevelMax_mAP=0.385.pth" \
    "https://zenodo.org/records/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1"
fi
if [ -f "weights/Cnn14_DecisionLevelMax_mAP=0.385.pth" ]; then
  echo "70539c43c18b6a289b3199c503a82c5a  weights/Cnn14_DecisionLevelMax_mAP=0.385.pth" | md5sum -c -
fi

# Trained models (GitHub release models-v0.5 of this private repo)
if ! (cd models && sha256sum -c --quiet SHA256SUMS 2>/dev/null); then
  rm -f models/*_best.pt models/SHA256SUMS
  gh release download models-v0.5 --dir models --pattern '*.pt' --pattern 'SHA256SUMS'
fi
(cd models && sha256sum -c SHA256SUMS)
echo "setup OK"
