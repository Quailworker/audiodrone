# audiodrone

Acoustic drone detection: a scenario simulator that renders what microphones hear while drones (and things that sound like them) fly past, and a CNN that detects drones in that audio frame by frame (10 ms frames).

- **ARCHITECTURE.md** is the map of the whole system: the stages, the code and data flow between them, where it runs, and how each part was built.
- **design.md** is the full design: theory, maths, decisions (D-01 to D-37), results (sections 5.9, 13.5).
- **AGENTS.md** is a step-by-step checklist for setting up the test environment and checking the published models.
- **NEXT_STEPS.md** is the to-do list: what to build next, in order, and how to start it on the desktop.
- **UPGRADES.md** lists where the best model falls short, ranked, and the planned upgrades.
- **results/real_world/REAL_TESTS.md** has the tests on public real drone and non-drone recordings, with suggestions to improve the model.
- **Live demo:** the UAE terrain tool (`terra/`) places mic clusters and drone flights on real UAE terrain in 3D and runs the whole chain. See [Live demo](#live-demo) below.
- **Border watch** (`/border`): the operator screen for the whole UAE land border. **Zetaris** connects the stored sensor, fusion and operations data, and **Nemotron through Meterless Nightshift** answers operator questions. See [section 10](#10-border-watch-zetaris-and-nemotron); `bash scripts/demo_check.sh` checks everything is up.
- **results/loc_v1/** has the Phase 1 localisation results (direction finding, tracking, tetrahedron vs flat clusters, real IDMT check); `tasks/phase1_localisation_REPORT.md` summarises them.
- This README covers how to replicate the published dataset, models and results.

Status (2026-10-25):
- **Border watch** is the operator screen: the 1,024 km land border, 3,395 clusters, and six simulated incidents replayed through the full chain, with "near the border" and "crossed the border" alerts.
- **Zetaris** runs locally in Docker and connects eight separate stores: six sector sensor stores, the fusion store (tracks and alerts) and the operations store (posts, shelters, units).
- **Nemotron 3.5 Lightning** answers through **Meterless Nightshift**, falling back to Nemotron Mini locally.
- **Simulation shows** that two-stroke drones are heard 1–1.4 km before the border and tracked before they cross. A quiet electric drone in the Hajar mountains or in Al Ain's town noise is heard only at or past the line.
- **Weakness:** tracks fragment into several IDs ("ghost tracks"), which shows up plainly in the Zetaris all-sectors view.

Earlier status (2026-10-24, later): the frequency tracker is rebuilt (`loc/freqtrack.py`, `results/freqtrack/`). It now guides the beamformer in place of the true shaft rates (chain v2), and tracking with it matches the oracle (`results/chain_v2/`). Live input from a sound card is in `live/` (section 9). Doppler in the tracker is built but off by default, because it made tracking worse on full scenes; `results/chain_v2/README.md` explains why and what to build instead. Cluster spacing study: `results/spacing/`.

Earlier status (2026-10-24): the UAE terrain tool runs the full chain (CNN detection, direction finding, tracking, alerts) on rendered sound over real terrain; model 2 (frequency-tracker input channels, `results/M2_logf_comb/`) trained but is not an upgrade at the standard threshold (fewer false alarms, shorter detection range), so model 1 (`B_logf`) stays the default.

Earlier status (design v0.9, models v0.5, 2026-10-23): Phase 1 localisation built and measured in simulation (design.md 13.5, `results/loc_v1/`). The CNN is trained on simulated audio only. It has been tested on public real recordings (detects real drones to ~100 m, but has many false alarms on everyday sounds; see UPGRADES.md). The main real test is the owner's own drone flyover recordings (design.md 14.7). Those are not in this repo and are never used for training.

## Live demo

The live demo is the **UAE terrain tool**. It runs in your browser:
1. Pick an area of the UAE.
2. Place mic clusters, a protected site and a drone flight on the 3D terrain.
3. Press Run. It renders the sound every mic would hear, then runs the real CNN, direction finding, the tracker and the alerts.
4. Play it back in 3D: the true drone against the tracked one, clusters lighting up when they hear it, bearings, line of sight (red when a hill blocks it), and the alert feed.

**You need:**
- Linux or Windows with Python 3.10+.
- An NVIDIA GPU is recommended. A CPU works, but the CNN step is slower.
- About 2 GB of disk.
- Internet for the first setup and for the page's 3D library.
- Read access to this repo's releases, for the model weights.

**Steps** (from the repo root; on Windows use `.venv\Scripts\python.exe`):

```bash
bash scripts/setup.sh                                                   # 1. Python env + model weights; ends with "setup OK"
.venv/bin/pip install -r requirements-alerting.txt -r requirements-terra.txt   # 2. web server (starlette/uvicorn via mcp) + rasterio
bash scripts/get_dem_uae.sh                                             # 3. UAE elevation tiles, ~870 MB, into data/dem (set DEM_DIR to put them elsewhere)
.venv/bin/python -m terra.server                                        # 4. start the tool, leave it running
```

5. Open **http://127.0.0.1:8790** in a browser.
6. Under **1 Area**, choose an **Example scenario**, then press **Run simulation**. "Over the ridge to a camp on the plain" is a gentle start; "Mountain site in the Hajar" shows hills blocking the sound. The progress log runs under the button, and a run takes about 30–70 s.
7. When the results band appears at the bottom, press **Focus** (top right), then **Play**. You can change the speed (1×, 4×, 10×) or drag the timeline. The **▶** buttons in the cluster table play what that cluster heard.

**Build your own scenario:**
1. Choose a preset area, or click the UAE map, then press **Load area**.
2. Under **2 Place**:
   - **Square of 5**, then click the ground: places five clusters at the set spacing.
   - **Site**, then click: places a protected site.
   - **Flight**, then click two or more waypoints: draws a drone flight. Press **New flight** to draw another.
3. Set the drone type, height above ground and speed under **3 Drone**, and the ambient sound and temperature under **4 Conditions**.
4. Press **Run simulation**.

Untick **Model the terrain** to rerun the same layout on flat ground and compare. Runs are saved in `runs_terra/` and reopen from **Previous runs**.

**Check it from the command line** (no browser): with the server running,

```bash
.venv/bin/python scripts/terra_check.py jais      # or: hafeet; prints the summary and alerts, ends with "terra check OK"
```

**If something goes wrong:**

| Problem | Fix |
|---|---|
| `address already in use` | Another program has port 8790: `python -m terra.server --port 8791` and open that port |
| "no DEM tiles … for this area" | Step 3 not run, or `DEM_DIR` points elsewhere. Tiles over open sea do not exist; pick a land area |
| Blank 3D view | The page loads three.js from cdn.jsdelivr.net; check the internet connection and the browser console |
| Run fails | The error shows in the progress log; the full traceback is in `runs_terra/<id>/error.txt` |

**Quicker text-only demos** (no terrain, no browser):
- `.venv/bin/python -m alerting.demo --no-llm` plays a scripted tabletop exercise through every alert level.
- `.venv/bin/python -m alerting.replay --pick 1 --no-llm` replays a simulated localisation scene. It needs the Phase 1 scenes, see section 6.

What the terrain model does and does not include is in [section 8](#8-uae-terrain-tool).

### Border watch (operator screen)

Open **http://127.0.0.1:8790/border** with the same server. It shows the whole UAE land border in 3D with a sensor line along it, and replays six simulated incidents through the real detection chain. It plays by itself, and flies to each incident as it happens. Section 10 covers the data behind it, Zetaris and the Ask box.

The screen works on its own. The ZETARIS drawer and the Ask box use Zetaris, Meterless and Nemotron when they are set up (section 10). Before a demo, run:

```bash
bash scripts/demo_check.sh --start    # starts what it safely can, checks the rest; ends with "READY" or what to fix
```

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
| `sim/layouts.py` | Mic cluster geometries (tetrahedron, flat) and field layouts (5 clusters on a 100 m or 150 m square, 3 in a line) |
| `loc/render.py` | Multi-cluster localisation scenes: 4-channel 8 kHz FLAC per cluster, ground-truth tracks and per-cluster audibility |
| `loc/doa.py` | Direction finding per cluster: pairwise SRP-PHAT-β, pair-delay least squares, reflection-aware beamformer (torch) |
| `loc/run_doa.py` | Runs direction finding (and the experiment variants) on every rendered scene |
| `loc/track.py` | Extended Kalman filter on cluster directions: emission-time correction, gating, track birth and association |
| `loc/evaluate.py`, `loc/report.py` | Scoring against ground truth; tables and figures in `results/loc_v1/` |
| `loc/idmt.py` | Direction finding on the real 8-channel IDMT recordings vs GPS |
| `terra/dem.py` | UAE terrain: Copernicus DEM tiles to a local metre grid, heights, ground slope, line of sight and knife-edge screening |
| `terra/scenario.py` | Renders a hand-built scenario on the terrain (cluster heights, sloped-ground echo, hills blocking sound), clusters in parallel |
| `terra/pipeline.py` | Runs the chain on a rendered scenario: CNN, beamformer, tracker over terrain, alert core; packages the result |
| `terra/server.py`, `terra/static/` | The web tool (Starlette + three.js), see [Live demo](#live-demo) |
| `terra/examples/` | Example scenarios shown in the tool |
| `scripts/get_dem_uae.sh`, `scripts/terra_check.py` | DEM download; command-line check of the tool |
| `results/M2_logf_comb/` | Model 2 (log-f + whitened spectrum + harmonic-comb channels): training log and evaluations |
| `loc/freqtrack.py` | Frequency tracker: harmonic comb + Viterbi pitch per mic, fine pitch, Doppler between clusters (design 6, 13.1) |
| `loc/eval_freq.py` | Scores the frequency tracker against the true rotor rates; writes `<scene>.freq.npz` |
| `live/run.py` | Live input for the field test: sound card or replayed WAV, CNN + frequency tracker + bearing, records the audio |
| `scripts/run_chain_v2.sh` | Chain v2 evaluation on the flat-cluster scenes (tracker-guided beamformer, real trigger, Doppler) |
| `scripts/spacing_study.py` | Cluster spacing study on flat ground (results in `results/spacing/`) |
| `tests/` | pytest unit tests: direction finding, tracking (with Doppler), rendering, alerting, terrain, live input, frequency tracker, border watch and Zetaris |
| `terra/border.py`, `terra/data/border.json` | UAE land border (Natural Earth, 42 sectors) and the cluster deployment along it |
| `terra/static/border.*`, `terra/data/incidents/` | Border watch screen, and the six incident replays it plays |
| `scripts/border_incidents.py` | Renders the incidents on the terrain and runs them through the chain |
| `alerting/border.py` | Border monitor: "near the border" and "crossed the border" alerts from the tracks |
| `scripts/sensor_stores.py`, `scripts/border_ops.py` | The separate stores (sector sensors, fusion, operations), loaded and registered in Zetaris |
| `alerting/sitedata_zetaris.py`, `terra/federated.py`, `terra/ask.py` | Site data read through Zetaris; cross-store queries; the Ask box (Nemotron) |
| `deploy/zetaris/docker-compose.override.yml` | Our memory caps and localhost-only UI for the local Zetaris stack |
| `scripts/demo_check.sh` | Checks (and starts) everything the demo needs |
| `results/loc_v1/` | Localisation results: `summary.json`, `summary.md`, figures, `idmt/` |
| `alerting/` | System layer: `rating.py` (D-32), `danger.py` (13.3/13.6.4), `core.py` (fixed-logic alerts), `delivery.py`, `agent.py` (Nemotron), `sitedata.py` + `sitedata_mcp.py` (Zetaris slot), `config.py`, `replay.py`, `demo.py`, `eval_rating.py` |
| `results/system_demo/` | Alert timelines from replays and the tabletop, rating evaluation |
| `results/sim_v1_manifest.csv` | One row per scene (length, level, sources, label counts) to check a re-render |

Not in git (see `.gitignore`): `data/` (rendered scenes, 26 GB), `weights/` (CNN14 checkpoint), `models/` and `runs/` (trained weights), `runs_terra/` and `runs_border/` (tool and incident runs with audio). `terra/data/` is in git.

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

## 6. Localisation (Phase 1)

Multi-cluster scenes use the same random scenes as `sim_v1` (same physics, new seeds from 20,000,000, only seeds whose scene has a drone), heard by every mic of a field layout. The same seeds are used for every configuration, so configurations are compared on identical flights. Rendering with the default single mic is unchanged (checked bit-for-bit against `sim_v1`).

```bash
.venv/bin/python -m pytest -q tests/                       # direction finding and tracker unit tests
bash scripts/render_loc.sh tetra_pole_100 8                # 300 scenes (~7 GB, ~175 CPU-s each), then direction finding
bash scripts/render_loc.sh flat_ground_100 8
bash scripts/render_loc.sh tetra_ground_100 8              # optional
bash scripts/render_loc.sh tetra_pole_150 8                # optional
.venv/bin/python -m loc.run_doa --config tetra_pole_100 --comb --limit 100     # oracle comb-guided experiment
.venv/bin/python -m loc.run_doa --config flat_ground_100 --comb --limit 100
.venv/bin/python -m loc.evaluate --configs tetra_pole_100 flat_ground_100 tetra_ground_100 tetra_pole_150
.venv/bin/python -m loc.idmt                               # needs data/real/idmt (Zenodo 17525811)
.venv/bin/python -m loc.report
```

Direction finding needs a GPU for speed (about 4 s per scene on an RTX 3080, main methods only); it runs on the CPU too.

**Chain v2 (frequency tracker instead of the oracles).** It needs scenes with their audio kept (`loc/render.py --audio`; the flat set used is in `data/loc_ft`):

```bash
bash scripts/run_chain_v2.sh data/loc_ft flat_ground_100   # bf, oracle comb, tracker comb (+ .freq.npz), scoring, tracking
.venv/bin/python scripts/doppler_sweep.py                  # Doppler noise sweep (results/chain_v2/doppler_sweep.txt)
.venv/bin/python scripts/spacing_study.py --out results/spacing --workers 3 --chain v2   # cluster spacing, on flat terra scenes
```

Results and explanations: `results/freqtrack/README.md`, `results/chain_v2/README.md`, `results/spacing/README.md`.

## 7. System layer: alerts, Nemotron, site data

`alerting/` turns tracks into alerts (design.md 13.6):
- **Core** (no language model): rating, danger zone, tiers.
- **Delivery:** log, webhook, ATAK CoT, operator approval.
- **Nemotron layer:** follow-ups and operator Q&A through an OpenAI-compatible endpoint, with the site data over MCP.

```bash
.venv/bin/pip install -r requirements-alerting.txt          # openai, mcp
ollama pull nemotron-mini                                   # stand-in for Meterless Nightshift (Linux)
.venv/bin/python -m pytest -q tests/test_alerting.py
.venv/bin/python -m alerting.demo                           # tabletop: on course -> evacuate -> take cover -> all clear
.venv/bin/python -m alerting.replay --pick 3 --method bf_comb2k   # simulated scenes end to end
.venv/bin/python -m alerting.eval_rating --method bf_comb2k       # rating: drone vs false tracks
```

**Current setup:** Nemotron 3.5 Lightning 30B-A3B runs on Anna's PC under Nightshift's bundled llama.cpp server (`scripts/two_pc/run_nemotron_anna.cmd`). This desktop reaches it through an SSH tunnel (systemd user unit `nightshift-tunnel`). The endpoint and key are in `~/.config/audiodrone/alerting.env`, which is not in git and is read automatically by `alerting/config.py`.

**Use the real services:**
- **Meterless Nightshift:** set `LLM_BASE_URL=http://<nightshift-pc>:8088/v1`, `LLM_API_KEY=<key>` and `LLM_MODEL=<Nemotron model name in Nightshift>`.
- **Zetaris:** set `SITE_MCP="<command that starts its MCP server>"`, exposing the tools in `alerting/sitedata_mcp.py`.

`alerting/example_site.json` is a made-up site. Real personnel locations belong in Zetaris and never in git.

## 8. UAE terrain tool

`terra/` is the browser tool behind the [Live demo](#live-demo): pick an area, place mic clusters, protected sites and drone flights in 3D, then run the whole chain on rendered sound and play it back. Setup and use are in the Live demo section.

**What a run does:** renders the sound at every mic on the terrain, then runs the chain:

| Stage | What runs |
|---|---|
| Detection | The real CNN (model 1) on each cluster's reference mic |
| Frequency tracker | Harmonic tone and pitch per cluster |
| Direction | The beamformer, guided by the tracker's tone lines (chain v2; `"chain": "v1"` for the plain beamformer) |
| Tracking | The Kalman tracker, with the terrain as the ground (`"doppler": true` adds the opt-in pitch state) |
| Alerts | The alert core |

A 2–3 minute flight with 5 clusters takes about 30–70 s. Runs are saved in `runs_terra/<id>/` with the scenario, results, alerts and each cluster's audio.

**Terrain in the sound model** (`terra/scenario.py`):
- **Heights:** each cluster stands on the DEM ground. Flights either follow the terrain at a set height or fly at a fixed altitude.
- **Ground echo:** comes from the local ground slope under each cluster.
- **Screening:** hills and buildings between the drone and a cluster cut the sound by single knife-edge diffraction (ITU-R P.526). The loss depends on frequency and is capped at 25 dB (ISO 9613-2).

**Limits:**
- Only one diffracting edge is modelled per path, with no multi-ridge losses or reflections off hillsides.
- Copernicus is a surface model, so buildings count as terrain.
- Detection uses model 1.

## 9. Field test: live input

`live/run.py` runs the per-cluster front of the chain in real time on a microphone (or a recording played back at real-time speed): CNN, frequency tracker, and with a 4-mic cluster a bearing. **It records the raw audio while it runs**, so every flyover can be re-analysed later with newer models. Field recordings are test data only (D-25): never add them to training.

```bash
.venv/bin/python -m live.run --list                                    # capture devices
.venv/bin/python -m live.run --device hw:1,0 --channels 1 --record field/2026-10-25_pass1.wav
.venv/bin/python -m live.run --device hw:1,0 --channels 4 --cluster flat --record field/c0.wav
.venv/bin/python -m live.run --replay data/real/svanstrom/audio/DRONE_001.wav     # no mic: replay in real time
```

One line every 0.5 s (GPU) or 1 s (CPU):

```
     4.5s  DRONE [#############       ] 0.63  tone   173.8 Hz (score 16.8)  pitch +0.48 %/s
```

- `DRONE` and the bar: the CNN's drone probability (model 1); the flag shows at 0.5 or more.
- `tone`: the strongest harmonic family the frequency tracker holds, and its comb score (5 or more counts as a tone).
- `pitch`: the tone's trend over the last 2 s. During a pass it falls as the drone goes by (Doppler). On one mic this is only a supporting cue, because an RPM change looks the same.
- With `--channels 4 --cluster flat`: azimuth, elevation and coherence from the tracker-guided beamformer.

The same records go to a JSONL log next to the recording. Input comes through PortAudio (`pip install sounddevice` plus the system PortAudio library) when available, else `arecord` on Linux. If the analysis falls behind real time it skips updates but never stops recording. `--gain-db` adjusts the level reaching the CNN.

## 10. Border watch, Zetaris and Nemotron

**The screen** (`terra/static/border.*`, served at `/border`):
- **Theatre:** the 1,024 km UAE land border (Natural Earth, 42 sectors) and 3,395 flat mic clusters, in two rows 300 m and 900 m inside it, 600 m apart (`terra/border.py`).
- **Incidents:** six simulated incidents replayed on one watch clock. Each was rendered on the real terrain at a border point, heard by the deployed clusters there, and run through the full chain (`scripts/border_incidents.py`). The border monitor (`alerting/border.py`) adds "near the border" and "crossed the border" alerts.
- **Zoomed in:** clicking an incident opens a 3D replay with clusters, bearings, tracks, alerts and an after-action line. SIM TRUTH shows where the drone really was.

**How the data fits together.** Live tracking streams directly (cluster to tracker to alerts), because a Zetaris query takes about a second. Everything the system stores is kept in separate databases, as separate servers would keep it, and **Zetaris connects them**:

| Zetaris datasource | Database | Holds |
|---|---|---|
| `sensors_<sector>` (6) | `sensor-db` / `sector_<sector>` | That sector's clusters, and what each heard per second: CNN probability, tone, bearing |
| `fusion` | `sensor-db` / `fusion` | Tracks, alerts, incidents |
| `borderops` | `border-ops-db` | Posts, shelters, patrol and garrison units (demo data) |

The **ZETARIS** button opens questions answered by one query across these databases (`terra/federated.py`), with the SQL shown:
- all sectors at once;
- which clusters heard the drone first, and from what bearing;
- seconds heard against seconds tracked;
- units near the tracks.

The **Ask** box sends an operator question to Nemotron (`terra/ask.py`), with the situation at that moment of the replay. It answers from Zetaris data and shows which model answered and where the data came from. It uses Nemotron 3.5 Lightning **through Meterless Nightshift** when Nightshift answers quickly, otherwise Nemotron Mini through local Ollama.

**Meterless setup:** Nightshift runs on the Windows GPU PC. Its API is on `127.0.0.1:8088` there, reached from here through the SSH tunnel on `18088`. Its NVIDIA provider serves `nvidia/nemotron-3.5-lightning-30b-a3b`. In `alerting.env`:
- `LLM_BASE_URL=http://127.0.0.1:18088/v1`;
- `LLM_API_KEY` = Nightshift's proxy token (`%LOCALAPPDATA%\ai.meterless.runtime\proxy-token` on that PC);
- `LLM_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b`;
- `LLM_MAX_TOKENS=1500`;
- `LLM_EXTRA_BODY={"chat_template_kwargs": {"enable_thinking": false}}`. Without it, a short reply is all reasoning and comes back empty.

The NVIDIA provider is hosted, so questions and their facts leave the machine. For real personnel data, use Nightshift's local runtime instead. Names and numbers not found in the data are flagged.

**Set up Zetaris locally (Linux, about 30 minutes).** You need Docker with the compose plugin, Java 8+, about 15 GB of free disk, and the hackathon's Zetaris download:
1. Sign in to their registry with the `docker login …` command from the Zetaris Cloud "Run locally" page.
2. Download `zetaris-platform.zip` from the same page and unzip it into `~/zetaris/`.
3. Copy our safety override into that folder: `cp deploy/zetaris/docker-compose.override.yml ~/zetaris/zetaris-platform/`. It sets memory caps (about 10 GB in total, no swap) and keeps the web UI on localhost only, which matters because the bundle ships a default admin password.
4. Run `./preflight.sh`, then `docker compose pull`, then `docker compose up -d`, then `./verify.sh`. All 9 checks should pass.
5. Then, from this repo:

```bash
.venv/bin/pip install JayDeBeApi==1.2.3
# private settings (never commit): ~/.config/audiodrone/alerting.env
#   ZETARIS_USER=admin@zetaris.com  ZETARIS_PASSWORD=...  BORDER_OPS_DB_PASSWORD=...  SENSOR_DB_PASSWORD=...
docker run -d --name border-ops-db --network zetaris_default --memory 512m -e POSTGRES_USER=borderops \
  -e POSTGRES_PASSWORD="$BORDER_OPS_DB_PASSWORD" -e POSTGRES_DB=borderops -v border-ops-data:/var/lib/postgresql/data postgres:16-alpine
docker run -d --name sensor-db --network zetaris_default --memory 512m -e POSTGRES_USER=sensors \
  -e POSTGRES_PASSWORD="$SENSOR_DB_PASSWORD" -e POSTGRES_DB=fusion -v sensor-data:/var/lib/postgresql/data postgres:16-alpine
.venv/bin/python scripts/border_incidents.py          # the six incidents (~45 min); also writes runs_border/<id>/result.json
.venv/bin/python scripts/border_ops.py load && .venv/bin/python scripts/border_ops.py register
.venv/bin/python scripts/sensor_stores.py load && .venv/bin/python scripts/sensor_stores.py register
.venv/bin/python scripts/sensor_stores.py check       # row counts through Zetaris
```

Neither database publishes a port. They are reachable only on the Zetaris Docker network, which is how Zetaris reaches them by name. Without Zetaris, Border watch still runs: site data falls back to each incident's own post, and the ZETARIS drawer reports that it is unavailable.

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
