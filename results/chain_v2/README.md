# Chain v2: frequency tracker → tracker-guided beamformer → EKF

The question: can the chain run on its own signals instead of simulator truth? Two oracles used to stand in:

| Oracle (chain v1 evaluation) | Real replacement (chain v2) |
|---|---|
| `bf_comb`: beamformer weighted onto the drone's **true** harmonic lines | `bf_trk`: weighted onto the lines the **frequency tracker** found (`loc/run_doa.py --trkcomb`) |
| Trigger: the cluster measures when the drone is **truly** audible | `:real`: the cluster measures when the tracker holds a tone |

We also test Doppler in the EKF (`:dop`, pitch state in `loc/track.py`).

```
scripts/run_chain_v2.sh data/loc_ft flat_ground_100
python scripts/doppler_sweep.py
```

Data: 120 flat-cluster scenes (`data/loc_ft/flat_ground_100`: 5 flat clusters on a 100 m square, D-30),
rendered with `loc/render.py` and `--audio`. 71 have a single drone and 49 have two.

## Direction finding (80 scenes with direction ground truth, 136,631 cluster-frames)

| Method | Median error | p90 | > 10° | |
|---|---|---|---|---|
| `bf` (whole band) | 13.7° | 55° | 61% | |
| `bf_comb` (oracle lines, 100–860 Hz) | 4.4° | 25° | 27% | oracle |
| `bf_comb2k` (oracle lines, 100–2000 Hz) | 2.6° | 11° | 11% | oracle |
| **`bf_trk` (tracker lines, 100–860 Hz)** | **5.3°** | 90° | 36% | real |
| **`bf_trk2k` (tracker lines, 100–2000 Hz)** | **3.0°** | 50° | 22% | real |
| `ls`, `srp` | 13.5°, 13.8° | 55° | 61% | |

The tracker-guided beamformer gets most of the oracle's gain in the median. The difference is the tail:
p90 is 50–90° against 11–25°. These are frames where the tracker follows another tonal source, or a single
rotor of the drone, so the beamformer points at the wrong thing.

## Tracking (EKF, single-drone scenes)

Frames where at least 2 clusters hear the drone. Horizontal, position and height errors are medians in metres.

| Method | Coverage | Horizontal | Position | Height | Velocity median (m/s) | Latency median (s) | Never tracked | False-track fraction |
|---|---|---|---|---|---|---|---|---|
| `bf` | 0.40 | 8.5 | 19.6 | 13.7 | 3.5 | 9.9 | 45% | 0.60 |
| `bf_comb` (oracle) | 0.76 | 6.2 | 11.3 | 6.2 | 3.6 | 2.5 | 10% | 0.38 |
| **`bf_trk`** | **0.69** | **5.9** | **10.2** | **5.1** | **3.5** | 2.8 | 17% | 0.44 |
| `bf_trk:real` | 0.69 | 6.8 | 11.7 | 5.8 | 4.1 | **1.0** | **7%** | 0.69 |
| `bf_trk:dop` | 0.66 | 7.3 | 12.7 | 6.4 | 4.8 | 3.0 | 17% | 0.47 |
| `bf_trk:real:dop` | 0.66 | 8.0 | 14.3 | 7.5 | 5.3 | 1.3 | 8% | 0.70 |
| `bf_comb:dop` | 0.71 | 7.9 | 12.9 | 6.8 | 4.9 | 2.5 | 11% | 0.42 |

On two-drone scenes, `bf_trk` is again the best real method: horizontal error 8.0 m against 11.6 m for
`bf` and 11.8 m for `bf_comb`, with coverage of 0.23.

**Conclusions**

1. **`bf_trk` replaces the oracle comb.** It tracks as well as the oracle, slightly better in error and
   slightly worse in coverage, and roughly halves the plain beamformer's errors. It is the default in
   terra chain v2.
2. **The tracker trigger (`:real`) is faster but noisier.** It starts tracks after about 1 s instead of
   about 3 s, and misses fewer drones (7% against 17%). But about 70% of confirmed-track frames are then
   false tracks, because the tracker holds a tone in 61% of frames with no drone audible
   (`results/freqtrack`). In the system the CNN gates this. The tone should be a trigger only together
   with the CNN, never alone.
3. **Doppler in the EKF makes tracking worse. It is now off by default** (terra `scn["doppler"]`, opt-in).
   See below.

## Why Doppler hurts, and what would fix it

At the tuned `sigma_lnf = 0.003` (0.3% pitch noise), Doppler is worse in every class, including the
single-prop fixed-wing two-stroke, which has the cleanest pitch (`doppler_sweep.txt`):

| sigma_lnf | Horizontal (m) | Velocity median (m/s) | Velocity p90 (m/s) |
|---|---|---|---|
| no Doppler | 5.18 | 3.33 | 13.6 |
| 0.003 | 6.47 | 4.60 | 18.7 |
| 0.01 | 5.60 | 3.63 | 14.7 |
| 0.03 | 5.18 | 3.25 | 13.4 |

It only stops hurting at about 3%, and by then it carries almost no weight. The Doppler signal itself is
good: pairs of clusters locked on the same line give a range-rate difference to 0.67 m/s median
(`results/freqtrack`). The problem is the measurement model. The EKF has **one** ln f0 state per track, and
every cluster's tone must match it within 0.3%. In real scenes the clusters lock on different rotors, which
differ by 10–20% on a quad, or on different family members, and the heavy tail (pair p90 ≈ 1.1%) pulls the
state off. On the clean synthetic test (`tests/test_track.py`), where every cluster hears the same line, the
same filter halves the horizontal velocity error.

**Next design (not built):** use frequency-difference measurements (FDOA-like) between pairs of clusters
that are locked on the **same line identity**, as `loc/eval_freq.py` does when it scores Doppler pairs,
instead of a shared pitch state. A pair measures −c·ln(f_i/f_j) = ṙ_i − ṙ_j directly, so it needs no
emitted-pitch state. A wrong line identity then breaks only that pair, not the whole track.

## Caveats

- Same simulator for scenes and design. All of this is simulation, not field data (D-25).
- The `bf_trk` outputs were computed before the refine fix in `results/freqtrack/README.md`. At most 0.02%
  of their frames used a bad pitch. Tracking and the Doppler rows were rerun with the fixed caches.
- The direction sigma for each tracking method comes from its base method (`sigma_from` in
  `loc/evaluate.py`): 0.20 rad for `bf`, 0.061 for `bf_comb`, 0.067 for `bf_trk`.
