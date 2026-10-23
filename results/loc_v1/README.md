# Localisation results, v1 (Phase 1)

Direction finding and tracking for multi-cluster scenes, plus a real direction-finding check on the IDMT array recordings. Full write-up: **design.md 13.5**. Short report: `tasks/phase1_localisation_REPORT.md`. Reproduce: README.md section 6.

## Files

| File | Contents |
|---|---|
| `summary.json` | Everything `loc/evaluate.py` measures, per configuration:<br>• `direction`: overall, gated, by range, elevation, class, background, in-band SNR, near-horizon elevation error, paired comparisons;<br>• `tracking` per method: coverage, position, height and velocity errors, latency, false tracks, by distance and by number of clusters hearing; `bf_same_scenes` for paired comparisons |
| `summary_table.csv` | One row per configuration and method with the headline numbers |
| `comb_vs_band.json` | The oracle-comb experiment on identical frames |
| `ops_per_frame.json` | Multiply-adds per 0.4 s frame for each direction finder at 8 and 4 kHz (ESP32 question, design 9.6) |
| `idmt/idmt_summary.json`, `idmt/idmt_frames.csv.gz` | Real IDMT flights: per-frame estimated vs GPS azimuth, raw and heading-calibrated (leave one flight out) |
| `direction_error_by_method.png` | Median direction error: pairwise SRP-PHAT, pair-delay least squares, reflection-aware beamformer, per configuration |
| `direction_error_vs_range.png` | Beamformer direction error vs range, per configuration |
| `direction_error_comb_vs_band.png` | Whole band vs 2 kHz band vs the drone's own comb lines, same frames |
| `height_error_vs_distance.png` | Tracked height error vs horizontal distance to the nearest cluster |
| `example_track.png` | A typical tracked scene: top view and height vs time |
| `idmt_azimuth_error_vs_range.png` | Real azimuth error vs distance |

## Headline numbers

Directions are the median of audible frames that pass the coherence gate; gross means > 10°. Tracks are from single-drone scenes.

| Configuration | Beamformer direction | Comb lines to 2 kHz (oracle) | Tracked height (median) | Coverage |
|---|---|---|---|---|
| Tetrahedron, 2 m pole | 6.5°, 43 % gross | 1.4°, 11 % | 4.4 m | 50 % |
| Tetrahedron, on the ground | 4.9°, 31 % | n/a | 5.5 m | 54 % |
| Flat, on the ground | 12.0°, 56 % | 2.5°, 10 % | 12.6 m | 37 % |
| Tetrahedron, pole, 150 m field | 6.8°, 44 % | n/a | 4.4 m | 50 % |

Real IDMT azimuth, after heading calibration, 25–150 m: 8.5° median, 44 % gross.

All results are simulated except the IDMT row. Tracking uses oracle detection: a cluster contributes a direction only when the drone is audible there by the CNN label rule.
