# Phase 1 localisation: report

**Date:** 2026-10-23
**Ran on:** the Quail desktop (Linux, RTX 4060 Ti) and Anna's PC (Windows, RTX 3080).
**Task:** `tasks/phase1_localisation.md`, steps 0–6.
**Full write-up:** design.md 13.5 (v0.9). **Results:** `results/loc_v1/`.

## What was built

| Step | Built | Files |
|---|---|---|
| 1 | **Cluster layouts and field layouts:**<br>• tetrahedron (20 cm), and flat (the base triangle plus a centre mic);<br>• pole mount (2 m) or ground mount (0.03 m);<br>• fields `square5_100`, `square5_150`, `line3_100`.<br>**Multi-cluster rendering:** an optional `label_mics` path in `sim/scene.py`.<br>**Per scene:** each cluster's 4 mics as 8 kHz 24-bit FLAC; ground truth (tracks, per-rotor shaft rates, true direction at emission time, range, radial speed, per-cluster audibility and in-band SNR) | `sim/layouts.py`, `loc/render.py`, `scripts/render_loc.sh` |
| 2 | **Direction finding** (torch, GPU or CPU):<br>• SRP-PHAT-β;<br>• pair-delay least squares;<br>• a new reflection-aware beamformer;<br>• operations per frame for each;<br>• experiments: β, a 2 kHz band, 4 kHz audio, oracle comb lines | `loc/doa.py`, `loc/run_doa.py` |
| 3 | **EKF tracker** on directions:<br>• emission-time correction;<br>• tangent-plane measurements, with flat-cluster elevation noise;<br>• χ² gating;<br>• two-ray birth;<br>• per-cluster global-nearest-neighbour assignment;<br>• track merging and deletion | `loc/track.py` |
| 4 | **Scoring:** direction, height, position, velocity, coverage, latency and false tracks; paired comparisons | `loc/evaluate.py`, `loc/report.py` |
| 5 | **Real check** on the 8-channel IDMT recordings vs GPS (azimuth) | `loc/idmt.py` |
| 6 | design.md v0.9 (13.5, register, 15, 16, 17), README section 6, `results/loc_v1/README.md`, this report | |

**Tests:** `pytest` gives 22 passing tests:
- direction finding on synthetic plane waves (noise and harmonic comb, tetrahedron and flat, 8 vs 4 kHz, below the horizon);
- the tracker on perfect and noisy measurements, emission-time correction, and covariance consistency (NEES);
- layouts, truth geometry and the comb mask;
- the multi-mic render path, which matches the default path for the same scene.

**Checks:**
- **The default rendering is unchanged.** Six `sim_v1` test scenes, re-rendered after the change, are bit-identical (FLAC and labels).
- `scripts/smoke_test.sh` ends `smoke test OK`.
- ruff is clean.

## Headline numbers (simulated unless marked real)

| | Tetrahedron, pole | Tetrahedron, ground | Flat, ground | Tetrahedron, pole, 150 m |
|---|---|---|---|---|
| Direction, pairwise SRP-PHAT | 15.4° | 15.4° | 12.2° | 15.7° |
| Direction, reflection-aware beamformer | **6.5°** (43 % > 10°) | **4.9°** (31 %) | 12.0° (56 %) | 6.8° (44 %) |
| Direction, oracle comb lines to 2 kHz | **1.4°** (11 %) | n/a | **2.5°** (10 %) | n/a |
| Elevation error near the horizon (el < 15°) | 3.0° | 2.9° | 20.0° | 3.2° |
| Tracked height, median (beamformer) | 4.4 m | 5.5 m | 12.6 m | 4.4 m |
| Tracked position, median | 12.9 m | 14.0 m | 18.8 m | 13.2 m |
| Coverage when ≥ 2 clusters hear it | 50 % | 54 % | 37 % | 50 % |

**Comb lines vs whole band, tracking, on the same 35 scenes:**
- tetrahedron, pole: height 3.8 → 3.2 m, coverage 49 → 78 %;
- flat, ground: height 13.5 → 4.8 m, coverage 38 → 79 %.

**Real (IDMT), azimuth only, heading-calibrated leave-one-flight-out:** **8.5° median** at 25–150 m, 44 % > 10°.

## Answers to the task's questions

- **F4, direction accuracy:**
  - simulated, whole band: 5–6.5° (tetrahedra);
  - with comb-guided direction finding: 1.4–2.5°;
  - real azimuth: 8.5°.
  - The assumed σ_u = 0.02 (1.1°) is optimistic by 2–8×.
- **Tetrahedron vs flat (open questions 19, 29):** tetrahedra are better in every measure, most of all near the horizon. Flat clusters become workable only with comb-guided direction finding.
- **Single-cluster height clue (5.12.3):**
  - works with the tetrahedron (about 3° near the horizon), given a few seconds of persistence;
  - does not work with flat clusters (20°).
- **Spacing:** 100 m and 150 m perform the same in these scenes, because masked directions dominate the error. Re-test once comb-guided direction finding is in.
- **ESP32:**
  - **Fits on the node:** least squares and lookup SRP-PHAT need ~13–14 M multiply-adds/s.
  - **Needs the server:** the reflection-aware beamformer needs ~620 M/s.
  - The echo breaks exactly the pairwise methods on tetrahedra, so **mode A (raw audio to the server) suits tetrahedra**.
  - 4 kHz direction-finding channels give the same directions as 8 kHz.

## What failed, surprised or was skipped

1. **The ground echo broke pairwise direction finding on tetrahedra.**
   - On a pole: 3–15° noise-free and a −2.4° elevation bias in the scenes.
   - On the ground: the top mic loses its vertical phase.
   - Fixed by modelling the echo in a beamformer.
2. **Masking.** In 37–44 % of the frames where the CNN label says the drone is audible, the drone is below everything else in 100–860 Hz. That is the main error source, and the comb experiment shows how to fix it.
3. **Full PHAT** (β = 1) lost tones in noise. β = 0.5 is used instead.
4. **IDMT heading.** The array's heading in the IDMT data is rotated by a constant −154° relative to the GPS frame. It was calibrated leave-one-flight-out and reported both raw and calibrated. The GPS log has no altitude, so elevation and height could not be checked on real data.
5. **Skipped:**
   - **Doppler in the EKF (step 3b)** and the F2 rotor-line check. Velocity from directions only is ~4 m/s median.
   - **Distributed GPS-synced mics** were not simulated.
   - **Two-drone tracking** gets only basic support: one direction per cluster gives 20–26 % coverage.
   - **Tracking uses oracle detection** (the label rule), not the CNN.
6. **False tracks:** 46–69 % of confirmed-track time is on interferers. The D-32 rating has to screen these.
7. **Bugs found and fixed while reviewing** (design.md 15):
   - coherence normalisation;
   - variant comparisons were unpaired;
   - Joseph-form R mismatch;
   - σ lookup for variant names;
   - the example-track scene choice.

## Compute and storage

| Item | Amount |
|---|---|
| Rendering | 1,200 scenes, ~175 CPU-s each (~58 CPU-hours over both PCs) |
| Scene data | ~21 MB per scene, not in git:<br>• **big boy storage** (`data/loc_v1`): 12 GB, holding tetra_pole_100, scenes 0–199 of tetra_pole_150, and the truth and direction files for all configurations;<br>• **Anna's PC** (`C:\Users\pc\Joseph AI\audiodrone\data\loc_v1`): 14.5 GB of audio for tetra_ground_100, flat_ground_100 and scenes 200–299 of tetra_pole_150 |
| Direction finding | ~4 s per scene (main methods, RTX 3080); ~60 s with all variants on the shared 4060 Ti |
| Results in git | `results/loc_v1/`, 3.6 MB |

## Recommended next steps

**Before Phase 2:**
1. **Comb-guided direction finding:** the per-cluster comb tracker supplies the drone's lines, and the beamformer uses only those bins up to ~2 kHz. It is the biggest measured gain.
2. **Doppler in the EKF** for velocity (13.2, D-13, D-16), plus the F2 check with the stored per-rotor rates.

**Phase 2:**
3. **Likelihood rating (D-32):**
   - height and speed from these tracks;
   - use it to screen the false tracks;
   - tune it on `data/loc_v1` (true height and speed are stored).
4. **Danger zone (D-35)** from these tracks. The filter's covariance is consistent and slightly conservative, which suits an uncertainty corridor.

**Field:**
5. **Measure real σ_u** with our own tetrahedron and a GPS-logged drone, with heading and level calibration (open question 37).
