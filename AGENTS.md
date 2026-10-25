# AGENTS.md: set up the test environment, check the models and run the live demo

Instructions for a coding agent (or a person) to go from a fresh clone to verified results. Work from the repo root and run every step in order. Each step says what success looks like; stop and report if a step doesn't match.

- Steps 1–5 check the CNN models.
- Step 6 (optional) replicates training.
- Step 7 sets up and checks the live demo, the UAE terrain tool.
- Step 8 checks live input (the field-test tool).
- Step 9 sets up Border watch with Zetaris; step 10 connects Nemotron through Meterless Nightshift; step 11 is the pre-demo check.

**Start with ARCHITECTURE.md** for the map of the system: every stage, the code it lives in, the data passed between stages, the machines, and how each part was built.

Background reading, only if you need it: README.md (replication overview and the Live demo section) and design.md (5.8 build plan, 5.9 results, 14.6 simulator, 13.5 localisation, 13.6 alerts).

## Ground rules

- **Never train on real recordings.** Real recordings are for testing only (design D-25).
- Keep generated files in their usual places: `data/` (scenes), `models/` (trained weights and evaluations), `runs/` (new training runs). All three are git-ignored, so don't commit them.
- Run Python with `.venv/bin/python` (`.venv\Scripts\python.exe` on Windows) and modules with `-m` from the repo root, e.g. `python -m cnn.evaluate`.
- Lint changed code with `ruff check --select E,F,W,B --ignore E501 --exclude 'cnn/panns_*.py' .`
- Rendering is CPU-heavy. Limit workers on a shared machine (`render_*.sh <workers>`); each worker sets `OMP_NUM_THREADS=1`.
- `terra/` renders with `multiprocessing` in spawn mode. Any script that calls `terra.pipeline.run` or `terra.scenario.render` must sit under `if __name__ == "__main__":`, or every worker re-runs the script and the run hangs.
- Don't commit `data/dem/` (elevation tiles) or `runs_terra/` (tool runs, with audio); both are git-ignored.
- Model 1 (`B_logf_best.pt`) is the default detector everywhere, including the terrain tool. Model 2 (`results/M2_logf_comb/`) is an experiment, not a replacement (see Known issues).

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

## Step 7: the live demo (UAE terrain tool)

This needs Step 1 (environment and model weights). A GPU is recommended; on a CPU the CNN step of each run is slower.

```bash
.venv/bin/pip install -r requirements-alerting.txt -r requirements-terra.txt
bash scripts/get_dem_uae.sh            # ~870 MB into data/dem; set DEM_DIR=/elsewhere to store them elsewhere
.venv/bin/python -m pytest -q tests/test_terra.py
```

Success:
- `get_dem_uae.sh` prints `ok` for 28 tiles per resolution. The two all-sea tiles print `404` and are skipped; that is expected;
- `5 passed`. The Jebel Hafeet test is skipped, not failed, if the tiles are missing.

Start the server and leave it running (in the background, or in a second terminal):

```bash
.venv/bin/python -m terra.server       # serves http://127.0.0.1:8790 ; --port to change
```

Then, in another shell:

```bash
curl -s http://127.0.0.1:8790/api/examples | head -c 200; echo
.venv/bin/python scripts/terra_check.py hafeet
.venv/bin/python scripts/terra_check.py jais
```

Success:
- `/api/examples` returns JSON listing `hafeet` and `jais`.
- Each check prints its progress (rendering per cluster, CNN, direction finding, tracking and alerts), a summary and an alert list, then `terra check OK (<seconds> s)`.
- A run takes about 30–70 s on a GPU machine with 6+ cores.
- The runs are seeded, so on the reference machine the numbers are:

| | `hafeet` | `jais` |
|---|---|---|
| `first_take_cover_s` | 156.25 | 122.65 |
| `drone_enters_site_s` | 172.0 | 129.6 |
| `warning_s` | 15.8 | 6.9 |
| Terrain blocking | C5 blocked about 11% of the time | 58–82% per cluster |

Small differences from GPU arithmetic are fine. A missing take-cover alert, or no tracks at all, is not.

To show it to a person: open http://127.0.0.1:8790, choose an **Example scenario**, press **Run simulation**, then **Focus** and **Play** (README, Live demo).

If it fails:

| Problem | Fix |
|---|---|
| `address already in use` | Port 8790 is taken: start with `--port 8791` and pass `--url http://127.0.0.1:8791` to `terra_check.py` |
| `no DEM tiles in … for this area` | `get_dem_uae.sh` not run, or `DEM_DIR` differs between the download and the server |
| Run status `error` | Read `runs_terra/<id>/error.txt` for the traceback |
| The run never finishes and many `spawn_main` processes keep restarting | A driver script without the `__main__` guard (Ground rules); stop it and add the guard |

## Step 8: live input check (field-test tool)

```bash
.venv/bin/python -m pytest -q tests/test_live.py
.venv/bin/python -m live.run --replay data/real/svanstrom/audio/DRONE_001.wav --speed 0 --log /tmp/live.jsonl
```

Expected on `DRONE_001.wav` (Svanström et al. 2021, CC0, Zenodo 5500576, `Data/Audio` unpacked to `data/real/svanstrom/audio`): the CNN flags drone from about 1.5 s on, and the tracker holds a tone near 172–175 Hz from about 3.5 s. With a microphone, `--list` shows the devices; always pass `--record` in the field so the audio is kept (test data only, never training, D-25). See README section 9.

## Step 9: Border watch, with Zetaris

The screen runs without Zetaris: open `http://127.0.0.1:8790/border` with the Step 7 server. Without Zetaris, site data falls back to each incident's own post, and the ZETARIS drawer reports that it is unavailable. The border geometry and the six incident replays are in git (`terra/data/`).

**Zetaris (local Docker, Linux).** You need Docker with the compose plugin, Java 8+, about 15 GB of free disk on the Docker filesystem, and the hackathon's Zetaris download.
1. Install: `sudo apt-get install -y docker.io docker-compose-v2 openjdk-17-jre-headless && sudo usermod -aG docker $USER`. Until you log out and in again, wrap Docker commands as `sg docker -c "…"`.
2. Sign in to the registry with the `docker login zetarishackathon.azurecr.io …` command from the Zetaris Cloud "Run locally" page. It carries a credential: don't paste it into logs, chats or the repo.
3. Unzip `zetaris-platform.zip` to `~/zetaris/`, then `cp deploy/zetaris/docker-compose.override.yml ~/zetaris/zetaris-platform/` (memory caps, UI on localhost only).
4. In `~/zetaris/zetaris-platform`, run `./preflight.sh`, then `docker compose pull`, then `docker compose up -d`, then `./verify.sh`. Success: `All 9 checks passed`.

**Our stores, then Zetaris on top** (README section 10 has the two `docker run` lines for `border-ops-db` and `sensor-db`):

```bash
.venv/bin/pip install JayDeBeApi==1.2.3
.venv/bin/python scripts/border_ops.py load && .venv/bin/python scripts/border_ops.py register
.venv/bin/python scripts/sensor_stores.py load && .venv/bin/python scripts/sensor_stores.py register   # needs runs_border/ (scripts/border_incidents.py)
.venv/bin/python -m pytest -q tests/test_border.py
.venv/bin/python scripts/sensor_stores.py check
```

Success:
- `border_ops.py load` reports 47 posts, 84 shelters and 47 units;
- `sensor_stores.py load` reports 135 clusters, 32,768 detection rows, 2,654 track rows and 120 alerts;
- `register` lists `sensors_mus02 … sensors_sau15` and `fusion`;
- the tests pass;
- `check` prints row counts for six `sensors_*.detections` tables and `fusion.incidents: 6`, all read through Zetaris.

Zetaris's SQL dialect (this build) uses:
- `CREATE DATASOURCE <name> DESCRIBE BY "<letters, digits, _ . - , only>" OPTIONS (jdbcdriver, jdbcurl, username, password)`;
- then `REGISTER DATASOURCE TABLES FROM <name>`;
- then query `<name>.<table>`.

The open-source Lightning syntax (`REGISTER … JDBC DATASOURCE … NAMESPACE`) does not parse.

Ground rules:
- Passwords and tokens live only in `~/.config/audiodrone/alerting.env` (mode 600). Never print or commit them.
- Neither database publishes a port. Zetaris reaches them by name on the `zetaris_default` Docker network.
- Keep the memory caps in `docker-compose.override.yml`.
- Live tracking never goes through Zetaris: a query takes about 1 s.

## Step 10: Nemotron through Meterless Nightshift

Nightshift runs on the Windows GPU PC. Its Linux build is ARM64-only. Its API listens on `127.0.0.1:8088` there; from the Linux desktop it is `127.0.0.1:18088`, through the `nightshift-tunnel` SSH tunnel.
1. On the GPU PC, open Nightshift and add an NVIDIA provider, or choose a local model under Setup ▸ Runtime.
2. Copy its proxy token (`%LOCALAPPDATA%\ai.meterless.runtime\proxy-token`) into `alerting.env` without printing it.
3. Set in `alerting.env`:
   - `LLM_BASE_URL=http://127.0.0.1:18088/v1`
   - `LLM_API_KEY=<the token>`
   - `LLM_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b`
   - `LLM_MAX_TOKENS=1500`
   - `LLM_EXTRA_BODY={"chat_template_kwargs": {"enable_thinking": false}}`

Without the last setting, Nemotron spends a short reply on reasoning and Nightshift returns `empty_response`. `GET /v1/models` with the token lists what Nightshift offers.

The fallback is Nemotron Mini through local Ollama (`ollama pull nemotron-mini`). The Ask box uses it when Nightshift does not answer within 3 s.

## Step 11: before a demo

```bash
bash scripts/demo_check.sh --start
```

It starts the terra server and the containers if they are down, then checks Zetaris, the federation, the tunnel, a Nemotron reply through Nightshift and the fallback. Success: every line `OK` and the last line `READY: open http://127.0.0.1:8790/border`. Each `FAIL` line says what to do.

## Known issues (design.md 5.9.4)

- Electric drones are only detected about 76 % of the time at 0–25 m in sim. This is probably the metric counting ignored or masked frames; under investigation.
- `drone_engine` fires on ground two-strokes (chainsaws, blowers): about 28 false alarms per hour on drone-free scenes (design D-28, open). For "any engine", use max(drone_engine, ground_engine).
- Real recordings (results/real_world/REAL_TESTS.md): real drones are detected out to ~100 m, but everyday non-drone sounds (voices, animals, horns, sirens) trigger the electric head on ~80 % of clips. Expect false alarms in the field. See UPGRADES.md for the planned fixes.
- Model 2 (frequency-tracker input channels, `results/M2_logf_comb/`): at the standard 0.5 threshold it has fewer false alarms than model 1 (`drone_engine` 10 vs 28 per hour on `test_sim`), but it also detects less at every range (2–10 points lower). The overall F1 is about the same. It has not been compared at a matched false-alarm rate yet, so model 1 stays the default.
- Terrain tool limits:
  - Only single knife-edge screening is modelled, with no multi-ridge loss or reflections off hillsides.
  - Copernicus is a surface model, so buildings count as terrain.
  - Flat clusters give compass bearings (median 4–30° error per cluster in the examples) but poor elevation (about 45° estimated against 15° true). The tool therefore draws bearings level.
  - With clusters 100–300 m apart, a drone more than about 500 m out often gets several short tracks and late or wrong "over it now" alerts.
- Border watch (simulated incidents):
  - **Ghost tracks:** one drone becomes several track IDs (Liwa 22, Ghuwaifat 12), some at implausible heights. The border monitor reports each intruder once per sector, but the Zetaris all-sectors view and the event log show the fragments.
  - **No early warning in two places:** a quiet electric drone was heard only at the border in the Hajar mountains (Hatta) and only 0.8 km inside in town noise (Al Ain). The two-stroke incidents were heard 1–1.4 km out and tracked before they crossed.
  - The NVIDIA provider in Nightshift is hosted, so Ask questions and their facts leave the machine. That is fine for the simulated demo data; real personnel data needs Nightshift's local runtime.
