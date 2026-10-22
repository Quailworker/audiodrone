# Potential areas for upgrades

Where the current best model (B: CNN14 + log-frequency front end, `models/B_logf_best.pt`) falls short, ranked worst first, with the evidence and what to change. Written 2026-10-23 from the failure analysis in design.md 5.10 and the real-recording tests in [results/real_world/REAL_TESTS.md](results/real_world/REAL_TESTS.md).

## How it was tested

| Test | What | Code |
|---|---|---|
| Error breakdown | 700 held-out sim scenes (test_sim 300, test_heldout 400); every error sliced by ~20 factors (distance, RPM, background, interferer, wind, drones in scene, ...) | `analysis/dump.py`, `frames.py`, `slices.py` |
| Recording effects | 21 perturbations of the 300 test_sim scenes: gain, clipping, filters, codecs, mic tilt, preamp noise, wind, hum | `analysis/robust.py` |
| Controlled sweeps | 868 new scenes, one variable at a time: range, RPM, each confuser alone, each background alone | `analysis/sweeps.py` |
| Real drones | Fraunhofer IDMT (1.36 h, GPS distance) and Svanström et al. (30 drone clips) | `analysis/real.py` |
| Real non-drone sounds | ESC-50 (2,000 clips, 50 classes), Svanström helicopter and background clips (60) | `analysis/real.py` |

Real recordings are used for testing only, never for training (design D-25).

## Ranked weaknesses

| # | Weakness | Evidence |
|---|---|---|
| 1 | **False alarms on real non-drone sounds (critical)** | The electric-drone head fires (≥ 0.5 s event) on **76–82 % of ESC-50 clips**, at 0, −20 and −40 dB gain alike: crows, babies, laughter, sirens, sheep, coughing ≈ 100 %; rain 2 %. On Svanström outdoor backgrounds with no motor, 87 % of clips fire and 36 % of frames are flagged. Keeping only content above 1 kHz makes 98 % of ESC-50 clips fire, and occlusion finds no single responsible band |
| 2 | **Compressed audio and small mics** | Electric F1 0.924 → 0.773 with low-bitrate MP3 (false positives 0.75 % → 3.07 %); → 0.721 with a 400 Hz high-pass, as small MEMS or phone mics apply (false positives 6.93 %); 55 dBA preamp noise → 0.754 |
| 3 | **Moving engines read as engine drones** | With no engine drone present, drone_engine fires on light aircraft 14.4 %, mopeds 13.5 %, mowers 6.7 % and helicopters 4.5 % of frames; chainsaws only 0.2 %. Real helicopters (Svanström): 38 % of frames flagged as some drone |
| 4 | **Masking** | Recall 56 % with a 65–75 dBA background; 77 % when another source is > 20 dB above the drone in band; ~54 % when an aircraft or ground engine is the loudest source |
| 5 | **Real range ≈ 150 m (Phantom 4 class)** | Real flights, any-drone frames: 99 % ≤ 25 m, 96 % 25–50 m, 82 % 50–100 m, 54 % 100–150 m, 3 % 150–200 m, 0 % beyond. The Mavic drops off sooner (29 % at 50–100 m) |
| 6 | **Big, slow electric drones look like engines** | Mid-class drones at 4,000–7,500 RPM: 65–91 % detected, up to 62 % of frames called "engine". Real: 7 of 30 Svanström drone clips (all electric) are mostly called "engine", 1 of them only engine |
| 7 | **Two drones at once** | Recall 80.5 % vs 92.6 % with one drone |
| 8 | **Over-confident probabilities** | Best F1 thresholds are ≈ 0.7 for the drone heads and 0.9 for ground, not 0.5 (AP: electric 0.947, engine 0.973, ground 0.969) |
| 9 | **Very close range in sim (< 10 m)** | 54.5 % over 7 scenes; the inspected case was a chainsaw masking the drone. The headline "76 % at 0–25 m" was mostly a metric artefact: 90–100 % on frames labelled audible |

## What already works

- **Real drones are detected with zero real training data.** Any-drone is flagged on 96–99 % of IDMT frames within 50 m and 91 % of Svanström drone frames; 30/30 Svanström drone clips produce an event. It fires 0 % at the IDMT site when the drone is beyond 200 m, so it isn't just reacting to that site.
- **Log-frequency beat mel on real drones**, e.g. 0.94 vs 0.77 on the hovering Phantom 4, and it separates drone clips from real background better (clip AUC 0.84 vs 0.77). Mel rejects real helicopters better (22 % vs 38 % of frames flagged).
- FPV drones are detected from 6,500 to 25,700 RPM, well beyond the training range.
- Insensitive to gain, clipping, resampling, 5 ms time shift and high-bitrate MP3.
- Pure noise, silence, hum and the simulated backgrounds alone almost never trigger it.

## Upgrades, in order of value

| Priority | Upgrade | Fixes | Effort | Needs a decision? |
|---|---|---|---|---|
| 1 | **Many more varied negatives in training.** Real non-drone sound (voices, animals, alarms, water, clicks, ambience) mixed into scenes as backgrounds and confusers, or many more synthetic negative families. Candidates with clean licences: FSD50K (CC BY / CC0 clips only), own ambient recordings | #1, part of #3 | Medium: a data loader for external clips plus a retrain (~6 h on Anna's 3080) | **Yes**: open question 21 / D-25 (real audio as training *negatives*) |
| 2 | **Hard-negative test gate.** Before any model ships: ESC-50 event rate < 10 % and Svanström no-motor background < 10 %, alongside the sim F1 | Stops #1 regressing | Small | No |
| 3 | **Recording-chain augmentation.** Random MP3/Opus round trips, high-pass 50–500 Hz, low-pass 4–16 kHz, preamp noise 30–55 dBA, mic tilt ±6 dB/decade | #2 | Small: augmentation code only | No |
| 4 | **Calibrate thresholds.** Pick per-head thresholds on val (≈ 0.7 / 0.7 / 0.9) or temperature-scale; ship them in the checkpoint | #8, part of #1 | Tiny | No |
| 5 | **Moving-engine negatives + decide D-28.** More drone-free scenes with mopeds, light aircraft, helicopters, mowers. If D-28 option 2 is chosen ("any engine" = max of the two engine heads), most of #3 stops mattering | #3 | Small | **Yes**: D-28 |
| 6 | **Oversample hard mixtures.** Drone under a loud tonal interferer, two drones, 65–75 dBA backgrounds | #4, #7 | Small: sampler weights | No |
| 7 | **Widen the mid class.** Lower tip speeds (4,000–7,500 RPM, 2-blade large props) so slow electric drones stop looking like engines | #6 | Small | No |
| 8 | **Field-validate range.** The user's own flyovers, with a flight log, give the real detection-vs-distance curve that sets cluster spacing | #5 | Field day | No |

Upgrades 2–4, 6 and 7 can go into one retrain. Upgrade 1 is the one that matters most and needs the D-25/Q21 decision first.

## Not finished

The 4,000-scene extra sim test set (`data/sim_eval`, on big boy storage) was only 6 % predicted when the run was stopped for low memory. The results above don't depend on it.
