# AGENTS.md: set up the test environment and check the models

Instructions for a coding agent (or a person) to go from a fresh clone to verified results. Work from the repo root and run every step in order. Each step says what success looks like; stop and report if a step doesn't match.

Background reading, only if you need it: README.md (replication overview) and design.md (5.8 build plan, 5.9 results, 14.6 simulator).

## Ground rules

- **Never train on real recordings.** Real recordings are for testing only (design D-25).
- Keep generated files in their usual places: `data/` (scenes), `models/` (trained weights and evaluations), `runs/` (new training runs). All three are git-ignored, so don't commit them.
- Run Python with `.venv/bin/python` (`.venv\Scripts\python.exe` on Windows) and modules with `-m` from the repo root, e.g. `python -m cnn.evaluate`.
- Lint changed code with `ruff check --select E,F,W,B --ignore E501 --exclude 'cnn/panns_*.py' .`
- Rendering is CPU-heavy. Limit workers on a shared machine (`render_*.sh <workers>`); each worker sets `OMP_NUM_THREADS=1`.

## Step 1: environment

```bash
bash scripts/setup.sh                      # GPU machine
bash scripts/setup.sh --cpu --no-train-weights   # CPU-only, testing only
```

Success means all of the following:
- it prints `torch <version> cuda True` (or `cuda False` with `--cpu`);
- `weights/Cnn14_DecisionLevelMax_mAP=0.385.pth: OK` appears, unless you passed `--no-train-weights`;
- `B_logf_best.pt: OK`, `A_mel_best.pt: OK` and `C_small_best.pt: OK` appear;
- the last line is `setup OK`.

If it fails:

| Problem | Fix |
|---|---|
| `gh release download` fails | Run `gh auth login` with an account that can read `Quailworker/audiodrone`, then rerun |
| Zenodo download fails or md5 mismatch | Delete the file and rerun. The correct size is 327,428,481 bytes; a resumed partial download gives a wrong checksum |
| No Python 3.10 | 3.11 or 3.12 should also work; `setup.sh` falls back to `python3` |

## Step 2: smoke test (about 2 minutes, CPU is fine)

```bash
bash scripts/smoke_test.sh
```

Success:
- `checked 4 scenes, 0 mismatched`;
- a `smoke F1 {...}` line with all values above roughly 0.7. Only 4 scenes are scored, so the numbers are noisy;
- a CSV of detections, then `data/smoke/test_sim/p0000001.detections.csv` and `.png` listed;
- `smoke test OK`.

## Step 3: render the test splits

```bash
bash scripts/render_test.sh 8      # workers; ~10 min at 8 workers, ~3 GB
```

Success: the last line is `checked 700 scenes, 0 mismatched`. The scenes are in `data/sim_v1/test_sim` (300) and `data/sim_v1/test_heldout` (400).

A mismatch means the simulator code or a dependency version differs from the reference. Check that `git status` is clean, then `pip freeze` against `requirements.txt` (numpy and scipy matter most).

## Step 4: evaluate the three published models

```bash
bash scripts/test_models.sh cuda   # ~10 min on a GPU; "cpu" takes a few hours
```

Success: the table printed by `scripts/compare_results.py` ends in `MATCH`. Per-model outputs land in `models/eval/<run>/eval_<split>.json`.

Lines marked `F1!` or `FA!`, or `p_detect` lines, are outside tolerance. Usual causes:
- step 3 was skipped or only partly done (check the scene counts: 300 and 400);
- a different label version (the scene `.json` must have `"labels": "snr_lines2"`).

## Step 5: test on a real recording

```bash
.venv/bin/python -m cnn.infer path/to/recording.wav --model models/B_logf_best.pt --frontend logf --arch cnn14
.venv/bin/python -m cnn.infer path/to/recording.wav --model models/A_mel_best.pt  --frontend mel  --arch cnn14
```

Always pass the matching `--frontend` and `--arch` for each model:

| Model file | `--frontend` | `--arch` |
|---|---|---|
| `B_logf_best.pt` | `logf` | `cnn14` |
| `A_mel_best.pt` | `mel` | `cnn14` |
| `C_small_best.pt` | `logf` | `small` |

A mismatch fails while loading the weights, or for A vs B silently gives garbage.

Outputs go next to the recording:
- `<name>.detections.csv`: events of at least 0.5 s with probability ≥ the threshold (0.5 by default);
- `<name>.detections.png`: the spectrogram and the three probability traces.

For a field flyover (design.md 14.7), report:
- whether the drone was detected;
- the first and last detection times compared with the flight log (which gives the detection range);
- any detections in the no-drone recordings (false alarms per hour).

## Step 6 (optional): replicate training

Only do this when asked. It needs a GPU, the full dataset and several hours:

```bash
bash scripts/render_dataset.sh 8   # 6,400 scenes, 26 GB; ends with "checked 6400 scenes, 0 mismatched"
```

Then run the three `cnn.train` commands in README.md section 4. While training:
- `runs/<run>/log.jsonl` gets a validation line every 1,000 steps;
- compare it with `results/<run>/log.jsonl`. Final val mean F1 should be about B 0.92, A 0.90, C 0.89 (±0.01; training is not seeded).

Then evaluate and compare: `python scripts/compare_results.py runs results`. It expects `runs/<run>/eval_<split>.json`, which `cnn.evaluate` writes next to `best.pt`.

## Known issues (design.md 5.9.4)

- Electric drones are only detected about 76 % of the time at 0–25 m in sim. This is probably the metric counting ignored or masked frames; under investigation.
- `drone_engine` fires on ground two-strokes (chainsaws, blowers): about 28 false alarms per hour on drone-free scenes (design D-28, open). For "any engine", use max(drone_engine, ground_engine).
- Real recordings (results/real_world/REAL_TESTS.md): real drones are detected out to ~100 m, but everyday non-drone sounds (voices, animals, horns, sirens) trigger the electric head on ~80 % of clips. Expect false alarms in the field. See UPGRADES.md for the planned fixes.
