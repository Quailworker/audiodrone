# Task: Phase 1 — localisation in simulation

**For:** the Claude Code session on the Quail desktop (Linux, RTX 4060 Ti).
**From:** Joseph, via the planning session on quail-laptop, 2026-10-23.
**Goal:** build and measure the **localisation half** of the system in the scenario simulator: multi-cluster rendering, direction finding per cluster, and a tracker that gives position, **height**, velocity and uncertainty. Then measure how accurate it is, compare tetrahedral and flat clusters, and check direction finding on real array recordings.

Work through the steps in order. Each step lists what "done" looks like. **Stop and write down what happened if a step fails in a way the instructions don't cover**; don't guess around it.

---

## 0. Read first

1. `AGENTS.md`: ground rules, environment, how to run things. Follow it.
2. `design.md` sections, in this order:
   - **7.3–7.4** (regular tetrahedron, 20 cm edge, top mic 16.3 cm above the base, direction finding below c/(2a) ≈ 860 Hz)
   - **7.8** (flat ground clusters, D-30)
   - **8.2–8.4** (cluster spacing; real range ~100–150 m)
   - **10** (height: why azimuth alone can't give it; h = z_cluster + ρ·tan(el))
   - **13.1–13.3** (multi-cluster Doppler; EKF state, Jacobians, tuning; danger zone)
   - **5.12** (the likelihood rating, which needs height and speed from this work)
   - **14.6** (scenario simulator)
   - **16**, flaws **F2** (no single f₀ for a multirotor), **F4** (direction accuracy is assumed), **F7** (EKF gaps)
3. `sim/scene.py` (`sample_and_render`), `sim/acoustics.py` (`propagate_point_source`, `emission_time`), `sim/render.py`.

**Decisions you must not reopen** (Joseph's): train on simulated data only (D-25); all drones count (D-18); tetrahedron is adopted (D-07), flat clusters are a proposal to compare (D-30); Nemotron is never in the alarm path (D-34). If your results argue against one of these, report it; don't change it.

## Hard rules

- **Do not change existing outputs.** Rendering with the default `mics` must stay bit-identical to `sim_v1`. Do not reorder or add random draws on the default path. After touching `sim/`, run `bash scripts/smoke_test.sh` (must end `smoke test OK`) and re-render a handful of `sim_v1` test scenes and run `scripts/manifest.py verify` on them (0 mismatched).
- **Never train on real recordings**; real data is for testing only.
- **Never commit `data/`, `models/`, `runs/`** or large audio. Commit code, small results (JSON/CSV/PNG under a few MB), and docs.
- Lint: `ruff check --select E,F,W,B --ignore E501 --exclude 'cnn/panns_*.py' .`
- `git pull` before you start and before every push. Small, focused commits. End each commit message with:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
- Rendering is CPU-heavy. Measure with a pilot first (step 1.4) before scaling.

---

## Step 1: multi-cluster scenes

### 1.1 Cluster layouts
Add a way to describe **clusters** (not just single mics), e.g. `sim/layouts.py`:

- **Tetrahedron** (D-07): 4 mics, regular tetrahedron, edge `a = 0.20 m`. Base: equilateral triangle in the horizontal plane centred on the cluster centre; top mic at `a·√(2/3) ≈ 0.163 m` above the base centroid.
- **Flat** (D-30): the same 4 mics' footprint flattened into the ground plane (or a flat equilateral triangle + centre mic; document which).
- **Mounting height:** a parameter. Render both **ground** (base at ~0.03 m) and **pole** (base at 2.0 m). The existing default draws a random mic height; for clusters, use explicit heights so the geometry is exact.
- **Field layouts** (clusters around the origin, where trajectories are centred):
  - `square5_100`: 4 clusters at the corners of a 100 m square + 1 at the centre.
  - `square5_150`: same with 150 m.
  - `line3_100`: 3 clusters in a line, 100 m apart (a cheap baseline).
- `sample_and_render` takes `mics` as `(x, y, z|None)` with z = None meaning "the random scene mic height". Extend it **without changing the default path** (e.g. a new optional argument for explicit cluster mic positions). The random draws must happen in the same order as today.

### 1.2 Ground truth
For every drone in a scene, save its **true track**: control-rate times `t`, positions `p` (x, y, z), velocities, and its shaft-rate array (or at least per-rotor BPF over time). Save per cluster, per analysis frame (step 2's frame rate), the **true direction** from the cluster centre to the drone **at emission time** (use `acoustics.emission_time`), true range, true height and horizontal distance. Store in a separate truth file (e.g. `<scene>.truth.npz`) so existing files are unchanged.

### 1.3 Labels per cluster
Today `frame_labels` is computed for mic 0 only. Compute labels (drone audible / ignore / not) **per cluster**, using each cluster's reference mic. Scoring in step 4 should only count frames where the drone is audible at that cluster.

### 1.4 Pilot, then render
- Render **10 pilot scenes** with `square5_100`, tetrahedron, pole mount. Record CPU time per scene and disk per scene. (20 mics ≈ 20× the single-mic cost, roughly 80 s CPU per scene; check.)
- Storage: 20 channels at 32 kHz is large. Either store **24-bit multichannel FLAC for a small debug subset** (~20 scenes) and, for the rest, run step 2 inside the render worker and store only per-frame direction outputs + truth; or downsample a copy to 8 kHz for direction finding (it only uses < 2 kHz). Pick one, document it.
- Then render the evaluation set: **≥ 300 scenes** per main configuration, with seeds in a new range that cannot collide with `sim_v1` (e.g. prefix `L`, seeds 20,000,000+). Configurations (prioritise the first two):
  1. tetrahedron, pole, `square5_100`
  2. flat, ground, `square5_100`
  3. tetrahedron, ground, `square5_100`
  4. tetrahedron, pole, `square5_150`
- Script: `scripts/render_loc.sh <config> <workers>`, ending with a DONE line like the existing renderers.

**Done when:** pilot timings recorded; ≥ 300 scenes of configs 1 and 2 rendered; smoke test and manifest checks still pass.

---

## Step 2: direction finding per cluster (`loc/doa.py`)

- **Method:** SRP-PHAT (steered response power with PHAT weighting) over a direction grid, using all 6 mic pairs of a cluster. Grid: azimuth 0–360°, elevation −15° to 90°, about 1° steps (refine around the peak). Optionally refine with least squares on the 6 pair delays:
  ```
  τ_ij = (p_j − p_i)·u / c     →   solve for u, then normalise
  ```
- **Band:** 100–860 Hz for the 20 cm tetrahedron (alias limit c/(2a)). Make the band a parameter. Also record an experiment with the band up to 2 kHz to see how bad aliasing is.
- **Frames:** about 0.25–0.5 s windows, hop 0.1 s (10 Hz, matching the EKF).
- **Output per frame:** azimuth, elevation, unit vector u, and a **quality** score (e.g. peak-to-second-peak ratio, or peak height over the map median).
- **Tonal signals:** drones are mostly tones, which give repeating GCC peaks (F4). Check how often the peak is wrong; if it is a problem, try weighting toward the drone's broadband component, or combining several frames.
- **Unit tests** (`tests/` or `loc/test_doa.py`, run with pytest): a synthetic plane wave from a known direction (white noise and a harmonic comb) must be recovered within 1–2° at high SNR, for both tetrahedron and flat layouts.

**Done when:** tests pass; per-frame direction outputs exist for all rendered scenes.

---

## Step 3: tracking (`loc/track.py`)

Implement the EKF from design **13.2**:
- State `s = [x, y, z, vx, vy, vz]` (step 3a), measurements = per-cluster direction unit vectors `u = (p − c)/r`, Jacobian `∂u/∂p = (I − u·uᵀ)/r`.
- Constant-velocity motion with white-noise acceleration (`q_acc ≈ 6 (m/s²)²/s` to start).
- **Measurement noise** per frame from the DOA quality score (or a constant σ_u measured in step 4 first).
- **Gating** (F7): reject measurements whose Mahalanobis distance exceeds a chi-square threshold.
- **Emission-time correction** (F7): the sound arriving now left the drone r/c earlier; evaluate the measurement model at `p − v·r/c`.
- **Initialisation:** from the first frame where ≥ 2 clusters have a good direction: closest point between the two rays (midpoint), velocity 0 with large covariance.
- **Multiple drones:** 10 % of scenes have two. Start with single-drone scenes; then add simple association (global nearest neighbour on gated measurements) and report both.
- **Step 3b (optional, after 3a works):** add Doppler. State gains `f0`; measurement `f = f0·c/(c + u·v)`; Jacobians in design 13.2; `q_f0 ≈ 100 Hz²/s` (D-13). This needs a per-cluster frequency track of one harmonic (design section 6 functional / complex demodulation). **Beware F2:** a multirotor has one BPF per rotor, a few % apart; check whether different clusters lock onto different rotor lines, and report it. The ground-truth per-rotor shaft rates from step 1.2 let you check this directly.

**Done when:** tracks (position, velocity, covariance per 0.1 s) exist for all evaluation scenes; a synthetic test with perfect measurements recovers the true track.

---

## Step 4: score it (`loc/evaluate.py`)

Against the ground truth, on frames where the drone is audible at the relevant cluster(s):

1. **Direction error** per cluster: the angle between estimated and true u. Report median, 90th percentile and the fraction of gross errors (> 10°), broken down by range (0–25, 25–50, 50–100, 100–150, 150+ m), true elevation, drone class, background level, and tetrahedron vs flat. **The median near-horizon elevation error is the number that decides D-32's single-cluster height clue (design 5.12.3).**
2. **Track errors:** position, **height**, and velocity RMS and 90th percentile, against horizontal distance from the nearest cluster and against how many clusters hear the drone (1, 2, 3+).
3. **Coverage:** fraction of drone time with a valid track; time from first audibility to first track.
4. **Configurations compared:** tetrahedron vs flat (answers open questions 19 and 29), pole vs ground mount, 100 m vs 150 m spacing.
5. Figures (PNG): direction error vs range; height error vs horizontal distance; an example track plotted against truth in 3D or top view + height.

Write results to `results/loc_v1/` (JSON + CSV + PNG; small files only).

---

## Step 5: real check on IDMT array recordings (open question 25)

The Fraunhofer IDMT drone flights (Zenodo 17525811, CC BY 4.0) are **8-channel array** recordings with **GPS** positions. `results/real_world/REAL_TESTS.md` used only channel 0. Expected location: `data/real/idmt/MicArrayData30s/`. If it's not on this machine, download it (it's test data; never train on it).

1. Find the **array geometry** (mic positions) and the array's position/orientation in the dataset documentation. **If the geometry is not documented, stop this step and report that.**
2. Run `loc/doa.py` on the 8 channels with the right geometry and a band that respects that array's alias limit.
3. Compare estimated directions with the true direction computed from the GPS track (and the array position). Report direction error vs distance, as in step 4.1.

This is the **first real measurement of direction accuracy** (F4). Report it honestly, good or bad.

---

## Step 6: write it up

1. **`design.md`:** add section **13.5 "Localisation results (v0.9)"**, tagged `[C]`, with evidence tags *(simulated)* / *(real)*. Include the key tables and the answers to: F4 (direction accuracy, sim and real), D-07 vs D-30 (open questions 19, 29), whether the single-cluster height clue (5.12.3) works, and what spacing the tracker needs. Update the status block to **v0.9**, the decision register's status column where the evidence changes it (don't change decisions themselves; mark them "evidence: …"), section 15 (errors you found and fixed), and open questions (mark 19, 25, 29 answered or updated; add new ones).
2. **README.md:** add the `loc/` package and `scripts/render_loc.sh` to the layout table and a short "Localisation" section with the commands to reproduce `results/loc_v1/`.
3. **`tasks/phase1_localisation_REPORT.md`:** a short report: what was built, the headline numbers, what failed or was skipped and why, compute/storage used, and recommended next steps (Phase 2: likelihood rating D-32 and danger zone D-35 using these tracks).
4. Push everything (`git pull` first). Tag nothing; Joseph tags releases.

## Definition of done

- [ ] Default rendering unchanged (smoke test OK, manifest verify 0 mismatched on re-rendered sim_v1 scenes)
- [ ] Multi-cluster rendering with ground truth and per-cluster labels
- [ ] `loc/doa.py` with passing unit tests
- [ ] `loc/track.py` (directions; Doppler optional) with a passing synthetic test
- [ ] `results/loc_v1/` with direction, height and track errors for tetrahedron vs flat
- [ ] IDMT real direction-finding result, or a written reason it couldn't be done
- [ ] design.md v0.9 section 13.5, README update, REPORT file; all pushed; ruff clean
