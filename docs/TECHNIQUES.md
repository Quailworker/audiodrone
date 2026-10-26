# Techniques

Every technique in the system: the problem it solves, how it works, where the code is, how it is tested, and what it measured. Each section is self-contained. Numbers come from the repo's own results files. Everything is in simulation unless it says "real".

| # | Technique | Code |
|---|---|---|
| 1 | [Physics-based acoustic scene simulator](#1-physics-based-acoustic-scene-simulator) | `sim/` |
| 2 | [Transfer-learned drone detector (CNN14, log-frequency front end)](#2-transfer-learned-drone-detector) | `cnn/` |
| 3 | [Harmonic frequency tracker (comb score + Viterbi)](#3-harmonic-frequency-tracker) | `loc/freqtrack.py` |
| 4 | [Ground-echo-aware, tone-guided beamforming](#4-ground-echo-aware-tone-guided-beamforming) | `loc/doa.py`, `loc/run_doa.py` |
| 5 | [Multi-cluster extended Kalman tracker, with ghost suppression](#5-multi-cluster-extended-kalman-tracker) | `loc/track.py` |
| 6 | [Terrain acoustics on real elevation data](#6-terrain-acoustics-on-real-elevation-data) | `terra/dem.py`, `terra/scenario.py` |
| 7 | [Fixed-logic alerting: rating, danger zone, border monitor](#7-fixed-logic-alerting) | `alerting/` |
| 8 | [Data federation with Zetaris](#8-data-federation-with-zetaris) | `scripts/sensor_stores.py`, `terra/federated.py`, `alerting/sitedata_zetaris.py` |
| 9 | [Nemotron through Meterless Nightshift, with grounding checks](#9-nemotron-through-meterless-nightshift) | `alerting/agent.py`, `terra/ask.py` |
| 10 | [Listening-range model and build mode](#10-listening-range-model-and-build-mode) | `terra/build.py`, `terra/static/border.js` |
| 11 | [Live input for field tests](#11-live-input-for-field-tests) | `live/run.py` |
| 12 | [Evaluation tooling](#12-evaluation-tooling) | `scripts/`, `tests/` |

---

## 1. Physics-based acoustic scene simulator

**Problem.** There is very little labelled real drone audio, and none for the drone classes and conditions we need. So we train on simulation only (design decision D-25), and test on real recordings.

**Method** (`sim/sources.py`, `sim/acoustics.py`, `sim/scene.py`):
- **Electric multirotors:** each rotor's blade-pass harmonics plus motor whine. Rotor speeds come from flight dynamics: thrust balances gravity, acceleration and drag, and the rear rotors spin faster in forward flight.
- **Two-stroke engines:** firing pulses through a resonance, an optional muffler, misfires, and propeller tones.
- **Confusers:** chainsaws, blowers, mopeds, traffic, aircraft, fans, mains hum, insects, birds, and diffuse beds.
- **Propagation:** each sample's **emission time is solved exactly**, so Doppler and travel delay emerge physically, read with a fractional delay. Then 1/r spreading, a ground-reflection image source, and **ISO 9613-1 air absorption** per STFT frame.
- **Labels by audibility, not by loudness:** a frame is positive when at least two tone regions stand 6 dB or more above everything else (`frame_labels`, version `snr_lines2`). A drone can be clearly tonal at −20 dB broadband signal-to-noise.

**Tested:** scenes are deterministic per seed. `scripts/manifest.py verify` checks 700 test scenes render bit-identically (AGENTS.md steps 2–3).

**Result:** `sim_v1` holds 6,400 scenes (26 GB). Four drone classes: FPV 5–7", electric 7–12", two-stroke multirotor, two-stroke fixed-wing.

## 2. Transfer-learned drone detector

**Problem.** Detect drones in 10 ms frames, in noise, from simulated training data that must still transfer to real microphones.

**Method** (`cnn/model.py`, `cnn/train.py`):
- **Model:** PANNs **CNN14**, pretrained on AudioSet, with a new 3-class head: `drone_electric`, `drone_engine`, `ground_engine`.
- **Log-frequency front end:** a mel filterbank bent at 50 Hz, so it is log-spaced over the whole drone range: n_fft 4096 at 32 kHz, 64 bands from 50 Hz to 14 kHz. The input normalisation (bn0) is relearned for the new bands. With log-frequency bands, a change of rotor speed shifts the whole harmonic pattern without changing its shape.
- **Training:** masked binary cross-entropy (ignored frames don't count), then two stages: 1,500 steps on the head only, 15,000 steps on everything. Augmentation: gain, polarity and a random spectral tilt, to stand in for unknown microphones.

| Model | Validation mean F1 |
|---|---|
| **B: CNN14 + log-f (default)** | **0.914** |
| A: CNN14 + stock mel | about 0.90 |
| C: 1.2M-parameter CNN and GRU from scratch (embedded candidate) | about 0.89 |

**Real-world test** (`results/real_world/REAL_TESTS.md`; IDMT, Svanström and ESC-50 recordings, never used for training): it detects **real drones out to about 100 m**. But everyday sounds (voices, horns, animals) trigger the electric head on many clips, so false alarms are the main weakness, and `docs/UPGRADES.md` ranks the fixes.

## 3. Harmonic frequency tracker

**Problem.** Know *which* frequencies belong to the drone, without the simulator's truth, so direction finding listens to the drone and not the background.

**Method** (`loc/freqtrack.py`, `HarmonicTracker`; per cluster, every 0.1 s):
1. **Spectrogram:** 0.4 s Hann window, 0.49 Hz bins, 30–3,900 Hz.
2. **Whitening:** subtract a ±40 Hz running mean.
3. **Comb score** on a 0.1 % log grid of candidate fundamentals from 40 to 1,300 Hz:

   $$C(f_0)=\sum_{k=1}^{K}0.6^{\,k-1}\big[S(kf_0)-0.25\,S((k-\tfrac12)f_0)\big]$$

   The half-way term penalises candidates whose gaps are as loud as their teeth, which rejects single lines and noise.
4. **Viterbi path** over the candidates plus a "no tone" state, with a jump limit and switching costs.
5. **Second tone:** notch the first tone's harmonics and repeat.
6. **Refine:** a parabolic peak per harmonic, then a weighted least-squares f0. It is kept only within 0.3 % of the comb's pitch.

**Tested:** `tests/test_freqtrack.py`, including a regression test for a refine bug that produced −209 Hz (now bounded).

**Result** (120 flat-cluster scenes): the tone is **locked in 73 % of audible frames**, with **0.15 % median pitch error**. On a real Svanström drone clip it holds 172–175 Hz.

## 4. Ground-echo-aware, tone-guided beamforming

**Problem.** Microphones on the ground hear the drone and its ground echo at the same frequency, so pairwise time-delay methods are biased by 3–40°.

**Method** (`loc/doa.py`, `BeamformerDOA`): steered response power over a direction grid, with a steering vector that **models the ground echo**:

$$a_i(\mathbf u)=e^{jk(x_iu_x+y_iu_y)}\big(e^{jkz_iu_z}+R\,e^{-jkz_iu_z}\big),\quad R=0.75$$

The spectra are PHAT-β weighted (β = 0.5) and the beam is computed on the GPU in torch. **Tone guidance** (`loc/run_doa.py`, `tracker_weights`): only the frequency bins on the frequency tracker's harmonic lines are used.

**Tested:** `tests/test_doa.py`; a real check on the 8-channel IDMT array recordings against GPS.

| Direction error (flat clusters, 136k cluster-frames) | Median |
|---|---|
| Plain beamformer, whole band | 13.7° |
| **Tone-guided, 100–860 Hz / 100–2,000 Hz** | **5.3° / 3.0°** |
| (Oracle, using the true rotor speeds) | (4.4° / 2.6°) |
| **Real IDMT array, azimuth against GPS** | **8.5°** |

## 5. Multi-cluster extended Kalman tracker

**Problem.** Fuse directions from many clusters into drone tracks: position, height, velocity, and an honest uncertainty.

**Method** (`loc/track.py`, `Tracker`):
- **Model:** constant velocity. The state also carries ln f0 and its rate for optional Doppler.
- **Measurements:** compared in the tangent plane of the predicted direction, with an **emission-time correction** (the sound left the drone r/c seconds earlier). Flat clusters measure elevation poorly, so the elevation noise is scaled by 1/sin(el).
- **Association:** global nearest neighbour per cluster, with a χ² gate. Tracks are born where two rays nearly meet; duplicates are merged.

**Ghost suppression** (profile `TRACKER_V12`; design 13.8.5). With flat clusters, any two bearings cross somewhere, and height is poorly fixed, so one drone used to become many tracks. The fix:
- a new track needs a **third cluster's bearing to agree** within 4°;
- **no track more than 800 m above the ground**;
- duplicates merge by a rule that **adapts to the front end**:
  - one direction per cluster: two tracks within 150 m in plan are one drone, because a cluster can feed only one of them;
  - several directions per cluster: merge only what **no cluster can tell apart** (swarm-safe).

| Measured with `scripts/retrack_eval.py` | Extra tracks, before → after | Most at once |
|---|---|---|
| Two-drone test | 3.5 → **0.16** | 7 → 3 |
| Liwa border incident | 8.7 → **1.3** | 14 → 5 |

**Tested:** `tests/test_track.py` covers consistency checks, Doppler, ghost births, plan merging, and a **swarm test** (two drones 70 m apart stay two tracks).

**Doppler,** evaluated honestly: it halves the horizontal velocity error when every cluster hears the same tone line. On full scenes, clusters lock onto different rotors, so it is **off by default** (`results/chain_v2/README.md`).

## 6. Terrain acoustics on real elevation data

**Problem.** Hills and buildings block sound, and clusters stand on slopes.

**Method** (`terra/dem.py`, `terra/scenario.py`):
- **Elevation:** Copernicus GLO-30/90 surface model, so buildings count as terrain.
- **Clusters and flights:** cluster heights come from the elevation; flights follow the terrain, smoothed as an autopilot would.
- **Echo:** the ground echo reflects off the **local fitted ground plane**.
- **Screening:** **single knife-edge diffraction** (ITU-R P.526),

  $$v=h\sqrt{\tfrac{2f(d_1+d_2)}{c\,d_1d_2}}$$

  with the loss capped at 25 dB (ISO 9613-2) and updated 10 times a second, frequency dependent.

**Tested:** `tests/test_terra.py`. The defaults are bit-identical to flat-ground rendering.

**Result:** in the Hatta mountains, a quiet drone is heard only at the border line. This finding shaped the sensor-placement question (design Q51).

## 7. Fixed-logic alerting

**Problem.** Alerts must be predictable and testable, so no language model decides them.

**Method** (`alerting/`):
- **Rating** (`rating.py`): CNN probability, persistence, height and speed give "possible", "likely" or "ground source".
- **Danger zone** (`danger.py`): the probability that the predicted position falls inside a site's radius, a non-central χ² computed from the track covariance.
- **Tiers** (`core.py`): advisory, on course, take cover, and evacuate (needs approval).
- **Border monitor** (`border.py`): signed distance to the border (UAE on the left of each line). It reports "near the border" and "crossed the border" once per intruder and sector.

**Tested:** `tests/test_alerting.py`, `tests/test_border.py`.

**Result** (simulated border incidents): **two-stroke drones were heard 1–1.4 km before the border and tracked before they crossed.** A quiet electric drone in mountains or town noise was heard only at or past the line.

## 8. Data federation with Zetaris

**Problem.** In deployment, each sector server, the fusion server and operations each keep their own database. Operators need questions answered across all of them.

**Method:**
- **Zetaris runs locally in Docker** (`deploy/zetaris/`: memory caps, UI on localhost).
- **Eight separate Postgres databases** are registered as Zetaris data sources (`scripts/sensor_stores.py`, `scripts/border_ops.py`):
  - six `sensors_<sector>` stores: clusters, and what each heard per second;
  - `fusion`: tracks, alerts, incidents;
  - `borderops`: posts, shelters, units.
- **Cross-store questions** are answered by **one Lightning SQL query** each (`terra/federated.py`), shown with their SQL on the screen:
  - all sectors at once;
  - which clusters heard the drone first, and from what bearing;
  - heard against tracked;
  - units near tracks.
- **Site data for the language model** is served over **MCP backed by Zetaris** (`alerting/sitedata_mcp.py --zetaris`, `alerting/sitedata_zetaris.py`).
- **Design rule:** live tracking never goes through Zetaris, because a query takes about 1 s against a 10 Hz loop. Zetaris connects the *stored* data.

**Tested:** `tests/test_border.py` covers mapping rows to the local frame, the queries and the routes. `scripts/demo_check.sh` checks it end to end.

## 9. Nemotron through Meterless Nightshift

**Problem.** Operators ask questions in plain English, but a language model must never invent numbers or decide alerts.

**Method** (`alerting/agent.py`, `terra/ask.py`):
- **Model:** **Nemotron 3.5 Lightning**, called through **Meterless Nightshift**'s OpenAI-compatible API. Thinking is off, so short answers are not empty.
- **Grounding:** the alert core is rebuilt at the moment being asked about. Facts come from the core and **from Zetaris**: each post's nearest shelter, the units near the 3 most credible tracks, the clusters first to hear. The model also has tool access over MCP.
- **Checks:** every number, unit and name in the answer is checked against the facts and tool results; anything unsupported is flagged on screen.
- **Time limits:** a hard answer deadline (no silent retries), a facts-only fallback, and a local **Nemotron Mini** fallback when Nightshift is busy.

**Tested:** `tests/test_alerting.py` covers the number, unit and name guards, the timeout fallback, and unwrapping nested tool arguments.

**Result:** answers take 3–6 s through Meterless, and the screen shows which model answered and where the data came from.

## 10. Listening-range model and build mode

**Problem.** "How close must the microphones be?" needs a quick, honest answer before running a full simulation.

**Method** (`terra/build.py`): the range is where a typical drone of the class falls 14 dB under the background noise, after spreading and absorption:

$$L_1 - 20\log_{10}r - \alpha r = L_{\text{bed}} - 14\ \text{dB}$$

The 14 dB margin is **fitted to the full simulation**: the desert quad first heard at 0.9–1.2 km, and in town the quads went unheard beyond 200 m.

| Background | FPV | 7–12" quad | Petrol multirotor | Petrol fixed-wing |
|---|---|---|---|---|
| Quiet desert | 1,170 m | 860 m | 2,280 m | 2,610 m |
| Town | 260 m | 170 m | 700 m | 860 m |

**On screen** (`terra/static/border.js`):
- **3D listening domes:** true spheres, corrected for the relief scale.
- **Gap lines:** coloured by whether two clusters can hear across the gap.
- **One-click run:** the full chain with a run-time estimate, then a replay with plain-language captions.

**Tested:** `tests/test_build.py`. The ranges are checked against the full-simulation results.

## 11. Live input for field tests

**Method** (`live/run.py`): reads a sound card, or replays a WAV in real time. A **streaming polyphase resampler** (identical to offline resampling) feeds the CNN, the frequency tracker and, for a 4-microphone cluster, the tone-guided bearing. It always records the raw audio. An update takes 0.19 s of its 0.5 s budget on the GPU.

**Tested:** `tests/test_live.py`. **Real:** on a Svanström drone clip, the CNN flags the drone from 1.5 s.

## 12. Evaluation tooling

| Tool | What it measures |
|---|---|
| `cnn/evaluate.py` | Frame F1, detection against distance, false alarms per hour |
| `loc/evaluate.py`, `scripts/run_chain_v2.sh` | Direction and tracking error with and without the oracles |
| `scripts/retrack_eval.py` | Ghost tracks: re-runs only the tracker on saved runs |
| `scripts/spacing_study.py`, `scripts/swarm_abudhabi.py` | Cluster spacing; a 6-drone swarm in town noise |
| `scripts/doppler_sweep.py` | Doppler noise settings |
| `results/real_world/` | Real recordings (testing only) |
| `tests/` | **70 unit and integration tests** (`pytest -q`) |

**Honest limits, measured not assumed:**
- **Swarms:** two drones heard by the same clusters appear as one, because each cluster reports one direction (design Q40). In the 6-drone Abu Dhabi swarm, only the loudest drone was tracked.
- **Quiet drones:** a small electric quad is heard only within about 170 m in town noise.
- **Height** is coarse with flat clusters (design D-38).
- **No field test with a real drone yet.**
