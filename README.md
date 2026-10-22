# audiodrone

Acoustic drone detection: a scenario simulator that renders what microphones hear while drones (and things that sound like them) fly past, and a CNN that detects drones in that audio frame by frame (10 ms frames).

- **design.md** is the full design: theory, maths, decisions (D-01 to D-36), results (section 5.9).
- **AGENTS.md** is a step-by-step checklist for setting up the test environment and checking the published models.
- **NEXT_STEPS.md** is the to-do list: what to build next, in order, and how to start it on the desktop.
- **UPGRADES.md** lists where the best model falls short, ranked, and the planned upgrades.
- **results/real_world/REAL_TESTS.md** has the tests on public real drone and non-drone recordings, with suggestions to improve the model.
- This README covers how to replicate the published dataset, models and results.

Status (v0.5, 2026-10-23): trained on simulated audio only. It has been tested on public real recordings (detects real drones to ~100 m, but has many false alarms on everyday sounds; see UPGRADES.md). The main real test is the owner's own drone flyover recordings (design.md 14.7). Those are not in this repo and are never used for training.

## Repo layout

| Path | Contents |
|---|---|
| `sim/acoustics.py` | Propagation: emission-time solve, fractional delay, 1/r spreading, ground reflection, ISO 9613-1 air absorption, A-weighting, levels |
| `sim/sources.py` | Sound sources: electric rotors and motor whine, two-stroke engine and propeller, confusers (vehicles, aircraft, fans, hum, insects, birds, crickets), diffuse backgrounds, wind |
| `sim/scene.py` | Random scenes: drone classes, trajectories, RPM from flight dynamics, confusers, rendering, frame labels |
| `sim/render.py` | Parallel scene renderer (FLAC + labels `.npz` + metadata `.json` per scene) |
| `cnn/model.py` | Front ends (`mel`, `logf`), `DroneCNN14` (fine-tuned PANNs CNN14), `SmallCNN` (from scratch) |
| `cnn/data.py` | Frame targets and ignore mask from the labels, random 10 s training crops, augmentation |
| `cnn/train.py` | Two-stage training |
| `cnn/evaluate.py` | Frame P/R/F1, detection probability vs distance, false alarms per hour |
| `cnn/infer.py` | Run a model on a real recording, writes `.detections.csv` and `.detections.png` |
| `cnn/panns_*.py`, `cnn/PANNS_LICENSE` | Vendored PANNs model code (MIT) |
| `scripts/` | Setup, rendering, testing, manifest and comparison scripts (below) |
| `scripts/two_pc/` | The exact scripts used to render and train across the two PCs (Linux RTX 4060 Ti and Windows RTX 3080), kept as a record |
| `results/<run>/` | Reference results per model: `args.json`, `log.jsonl` (val metrics every 1,000 steps), `train.out`, `eval_test_sim.json`, `eval_test_heldout.json` |
| `analysis/` | Failure analysis (design.md 5.10): per-frame tables, slices, robustness, sweeps, real-recording tests, figures |
| `results/real_world/` | Real-recording results: summary (`REAL_TESTS.md`), figures, per-clip and per-range CSVs |
| `results/sim_v1_manifest.csv` | One row per scene (length, level, sources, label counts) to check a re-render |

Not in git (see `.gitignore`): `data/` (rendered scenes, 26 GB), `weights/` (CNN14 checkpoint), `models/` and `runs/` (trained weights).

## 1. Setup

Requirements: Linux (Windows works for rendering and training too: see `scripts/two_pc/*.cmd`), Python 3.10, an NVIDIA GPU for training (testing also runs on CPU), and the GitHub CLI (`gh`) logged in, with access to this private repo, for the trained models.

```bash
git clone https://github.com/Quailworker/audiodrone.git && cd audiodrone
bash scripts/setup.sh            # add --cpu for a CPU-only torch, --no-train-weights to skip the CNN14 download
```

`setup.sh` creates `.venv`, installs `requirements.txt` and fetches two sets of weights, checking both against their checksums:

- the PANNs checkpoint `weights/Cnn14_DecisionLevelMax_mAP=0.385.pth` (Zenodo record 3987831, md5 `70539c43c18b6a289b3199c503a82c5a`);
- the three trained models from the GitHub release `models-v0.5`:

| File | Model | sha256 |
|---|---|---|
| `models/B_logf_best.pt` | CNN14 + log-frequency front end | `9b1d2ae9d0e275db3eee412befdc9bbf409f990d1f63b46d078f9005e5d650ae` |
| `models/A_mel_best.pt` | CNN14 + stock mel front end | `22e0e096690c9daa5459061fb4936f05b57dfd069a21d3606851fb37e24c2320` |
| `models/C_small_best.pt` | small CNN trained from scratch | `5e8fed5e1d7428b233150f7a8b6b8946bdad35e5c94d572a6b1c5d96691ab180` |

## 2. Render the data

Rendering is deterministic per seed, and the seed alone sets the scene and its split: seed mod 20 = 0 is val, 1 is test_sim, anything else is train. Held-out scenes use `--heldout`, which adds 10,000,000 to the seed and switches to unseen conditions.

```bash
bash scripts/render_test.sh 8      # only test_sim + test_heldout: 700 scenes, ~3 GB, ~10 min with 8 workers
bash scripts/render_dataset.sh 8   # everything: 6,400 scenes, ~26 GB, ~1 h with 8 workers
```

Both scripts finish with `scripts/manifest.py verify`, which compares every rendered scene with `results/sim_v1_manifest.csv`. On Linux the FLAC files come out bit-identical to the reference. Between platforms the audio can differ by 1 LSB of 24-bit (−150 dB), so the check uses tolerances rather than file hashes. Expect `checked N scenes, 0 mismatched`.

| Split | Scenes | Seeds |
|---|---|---|
| train | 5,400 | `p` 0–3999 and `a` 1,000,000–1,001,999, seed mod 20 ∉ {0, 1} |
| val | 300 | same ranges, seed mod 20 = 0 |
| test_sim | 300 | same ranges, seed mod 20 = 1 |
| test_heldout | 400 | `ah` 0–399 with `--heldout` |

Each scene is `data/sim_v1/<split>/<name>.flac` (mic 0, 32 kHz, 24-bit, full scale 120 dB SPL), plus `.npz` frame labels and `.json` metadata (every random parameter, the sources and their levels, and the label version `snr_lines2`).

## 3. Test the published models

```bash
bash scripts/smoke_test.sh         # 2 min: renders 4 scenes, checks them, evaluates the small model, runs infer.py
bash scripts/test_models.sh cuda   # full test (needs render_test.sh first); use "cpu" without a GPU (a few hours)
```

`test_models.sh` evaluates B, A and C on test_sim and test_heldout, then runs `scripts/compare_results.py` against `results/`. It prints one line per model, split and class, followed by `MATCH`. The tolerances are F1 ±0.01, detection probability ±0.03 per range bin, and false alarms ±max(3/h, 15 %).

Reference results (design.md 5.9):

| Model | test_sim F1 electric / engine / ground | test_heldout F1 electric / engine / ground |
|---|---|---|
| B: CNN14 + log-f | 0.924 / 0.930 / 0.921 | 0.896 / 0.903 / 0.873 |
| A: CNN14 + mel | 0.922 / 0.912 / 0.926 | 0.913 / 0.898 / 0.850 |
| C: small, scratch | 0.883 / 0.834 / 0.890 | 0.875 / 0.820 / 0.845 |

## 4. Retrain

Needs the full dataset (`render_dataset.sh`) and the CNN14 checkpoint. These are the exact commands behind the published runs:

```bash
PY=.venv/bin/python
$PY -m cnn.train --data data/sim_v1 --out runs/B_logf  --frontend logf --arch cnn14 --batch 16 --workers 3
$PY -m cnn.train --data data/sim_v1 --out runs/A_mel   --frontend mel  --arch cnn14 --batch 16 --workers 3
$PY -m cnn.train --data data/sim_v1 --out runs/C_small --frontend logf --arch small --batch 16 --workers 6 --lr2 1e-3
# then, per run:
$PY -m cnn.evaluate --data data/sim_v1 --split test_sim test_heldout --model runs/B_logf/best.pt --frontend logf --arch cnn14
```

Defaults are stage 1 = 1,500 steps at lr 1e-3 (head, `bn0` and `fc1` only), then stage 2 = 15,000 steps with cosine decay at `--lr2` (default 1e-4) over everything except the fixed front end. Validation runs every 1,000 steps and the best mean F1 is saved as `best.pt`. Training is not seeded, so a retrain should land within about ±0.01 F1 of the reference rather than exactly on it.

Times on the hardware used:

| Run | GPU | Time |
|---|---|---|
| B | RTX 3080 at 240 W | 2 h |
| A | RTX 3080 at 240 W | 35 min |
| C | RTX 4060 Ti | 35 min |

GPU memory stays under 4 GB at batch 16.

## 5. Run on a real recording

```bash
.venv/bin/python -m cnn.infer recording.wav --model models/B_logf_best.pt --frontend logf --arch cnn14 [--channel 0] [--threshold 0.5]
```

Any sample rate is accepted; the audio is resampled to 32 kHz. The script writes:

- `recording.detections.csv`: one row per detected event, with class, start, end and peak probability;
- `recording.detections.png`: the spectrogram with each class's probability over time.

The level of real audio is not calibrated. That's fine, because training randomised gain by −25 to +15 dB.

## Outputs and labels

Per 10 ms frame, the model gives three independent probabilities:

| Output | Meaning |
|---|---|
| `drone_electric` | 5–12" electric multirotors |
| `drone_engine` | ~10"+ two-stroke drones, multirotor or fixed-wing |
| `ground_engine` | chainsaws, leaf blowers, mopeds, mowers |

A frame's label is positive when at least two separate tone regions of that source stand ≥ 6 dB above everything else at the mic. It's negative when no region reaches 0 dB, and ignored in between. The full rule is in design.md 5.8.6.

## Licences

`cnn/panns_models.py` and `cnn/panns_pytorch_utils.py` come from PANNs (Kong et al., 2020) under the MIT licence (`cnn/PANNS_LICENSE`). The CNN14 weights are downloaded from Zenodo record 3987831.

Real test recordings (not in git, testing only): Fraunhofer IDMT drone flights (Zenodo 17525811, CC BY 4.0), Svanström et al. 2021 (Zenodo 5500576, CC0), ESC-50 (CC BY-NC 3.0, non-commercial).
