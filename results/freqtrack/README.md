# Frequency tracker results

`loc/freqtrack.py` is the harmonic frequency tracker, rebuilt in October 2026 because the original code was
lost. It follows each cluster's strongest harmonic tone, or two tones, through time. The rest of the
system uses it three ways:

- as the trigger ("is there a tone here?");
- to guide the beamformer (`bf_trk` in `loc/run_doa.py`);
- for Doppler (the opt-in EKF pitch state in `loc/track.py`).

```
python -m loc.eval_freq --scenes data/loc_ft/flat_ground_100 --out results/freqtrack/flat_ground_100 --workers 3
```

Each scene's tracker output is cached as `<scene>.freq.npz`. `--force` re-runs the tracker.

## How it is scored

For every drone and every cluster, the received harmonic lines come from the simulator truth. A tracked tone
counts as **locked** when it is within 0.5% of a family fundamental: m × the blade-pass line, with
m ∈ {1–4, blades, 2·blades, 3·blades}.

| Metric | Meaning |
|---|---|
| detect | Fraction of audible frames (line ≥ 6 dB over the bed) where one of the two tracked tones is locked |
| pitch error | Median \|f/f_true − 1\| on locked frames, coarse comb grid and refined |
| tone rate, no drone audible | Fraction of frames with a tone where no drone is audible. Other tonal sources (generators, traffic, birds) make this high, so the tone alone is **not** a drone detector; the CNN is |
| Doppler pair error | For two clusters locked on the same line, −c·ln(f_i/f_j) against the true range-rate difference |

## Flat clusters, 120 scenes (`data/loc_ft/flat_ground_100`, D-30 layout)

`summary.json`, after the refine fix below:

| | Value |
|---|---|
| Audible frames | 184,859 |
| **Detect** | **73.3%** |
| By class | fpv 76.4%, mid 55.5%, ts_fw 84.4%, ts_mr 73.5% |
| By range | ≤50 m 80%, 50–100 m 75%, 100–200 m 73%, 200–400 m 74%, >400 m 69% |
| Pitch error | 0.16% coarse, 0.15% refined |
| Tone rate, no drone audible | 61% |
| Doppler pair error (vs kinematics) | median 0.67 m/s, p90 3.9 m/s |
| Doppler pair error by class | fpv 0.70, mid 0.68, ts_fw 0.53, ts_mr 0.92 m/s |

The tracker locks on the fundamental most often (m = 1: 96k frames), then on m = 2 and 3.

## Tuning (40 tetra_pole_100 scenes, before the flat run)

`HARM_DECAY`, `GAP_WEIGHT` and `NULL_SCORE` were picked on a separate set of 40 tetrahedral-cluster scenes,
not on the flat set above:

| GAP_WEIGHT | HARM_DECAY | NULL_SCORE | Detect | mid | Tone rate, no drone | Doppler (m/s) | Pitch error |
|---|---|---|---|---|---|---|---|
| **0.25** | **0.6** | **5.0** | **63.5%** | 26% | 52% | 0.77 | 0.14% |
| 0.25 | 0.75 | 6.0 | 50.2% | 10% | 38% | 0.69 | 0.14% |
| 0.25 | 0.6 | 6.0 | 58.5% | 19% | 45% | 0.74 | 0.14% |

A higher null score cuts tones where no drone is audible, but it loses the weaker drones first. The CNN does the
detecting, so the tracker is tuned to keep the drone.

## Weak spots

- **mid (7–12" electric quads), 55%.** Four rotors running 10–20% apart give four nearby harmonic families,
  and the comb splits its score between them (design §13.1, F2).
- **Tones from other sources, 61%.** Fine for direction guidance, because the CNN gates it. The `:real`
  trigger, which starts tracks on the tracker tone alone, pays for it in false tracks (results/chain_v2).

## Fix on 2026-10-24: refine gave nonsense pitches

The refined pitch, a parabolic peak per harmonic plus a weighted least-squares f0, was sometimes far
outside the comb's window: −209 Hz for a 178 Hz tone, in 30 of 1.1 million tracked frames. The peak bin
is picked on the whitened spectrum, so in the raw dB spectrum that bin can sit on a slope with almost zero
curvature, and the parabola offset blows up.

Fix:
- only interpolate at a real local maximum;
- clip the offset to ±0.5 bin;
- drop a refined pitch more than `REFINE_REL` (0.3%) from the coarse one, since every consumer falls back
  to the coarse pitch.

The regression test is `tests/test_freqtrack.py::test_fine_pitch_never_leaves_the_refine_window`. The numbers
above were rerun with the fix. The Doppler pair error went from 0.80 to 0.67 m/s median. The `bf_trk`
beamformer outputs were not rerun; at most 0.02% of their frames used a bad pitch.
