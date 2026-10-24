# Cluster spacing study (chain v2, flat ground)

```
python scripts/spacing_study.py --out results/spacing --workers 3 --chain v2
```

**Setup:**
- **Field:** five flat clusters, four on the corners of a square of side S plus one in the middle, with a
  protected site of radius 60 m at the centre.
- **Spacings:** S = 100, 300, 600, 1000 m.
- **Classes:**
  - 7–12" electric (`mid`): 60 m up, 15 m/s;
  - 5–7" FPV (`fpv`): 30 m up, 25 m/s;
  - two-stroke multirotor (`ts_mr`): 100 m up, 15 m/s.
- **Flights:** each class flies twice:
  - head-on at the site from 3 km;
  - a straight pass 400 m to the side (`offset`), which should never give take-cover.
- **Chain:** terra chain v2, with the tracker-guided beamformer and Doppler off.
- **Conditions:** desert ambient, seed 7.

Raw rows: `runs_v2.csv`. Alert logs: `runs/v2_<S>_<cls>_<flight>/alerts.jsonl`. The warning table below comes from `python scripts/spacing_warning.py`.

**Sample size:** one flight per cell (24 runs). Treat the numbers as indications, not statistics.

## Main finding: the alert rule, not the spacing, limits warning time

The CNN hears the drone, and the operators get "possible drone heard" advisories, with the drone
**900–3,000 m out**, 60–200 s before it reaches the site. But take-cover waits for a firm track rated
"likely". That only forms once the drone is near or inside the cluster field, so the **real warning is 0–18 s at
every spacing**:

| S | mid: warning | fpv: warning | ts_mr: warning | First advisory, drone distance (mid / fpv / ts_mr) |
|---|---|---|---|---|
| 100 m | 10.9 s | 0 s* | 14.8 s | 895 / 1139 / 2993 m |
| 300 m | 14.8 s | 3.8 s | 18.2 s | 908 / 1259 / 2993 m |
| 600 m | 8.9 s | −0.2 s | 14.8 s | 1028 / 1406 / 2996 m |
| 1000 m | 8.6 s | −0.3 s | 9.7 s | 1183 / 1444 / 2995 m |

"Warning" is site entry minus the first take-cover that was not a false "over it now".

\* At 100 m the FPV's first take-cover was 40 s early, but it was a false track (position near the site while the
drone was about 1 km out), so it is not counted. Three such false "over it now" alerts were sent.

Why the FPV gets no warning at 600 m and above: the first track, at 81 s, had the height badly wrong (362 m
against 30 m true; flat clusters give only coarse height, D-38). So it was never rated "likely", and take-cover
came only once the drone was over the middle cluster.

## What the spacing changes

| S | Distinct tracks per head-on pass (mid, fpv, ts_mr) | False take-cover on the 3 offset passes | False "over it now" alerts | Track error, electric classes (m) |
|---|---|---|---|---|
| 100 m | 4, 2, 3 | 1 | 4 | 31, 140, 82, 38 |
| 300 m | 3, 1, 4 | 1 | 0 | 11, 67, 1, 22 |
| 600 m | 4, 1, 3 | 0 | 0 | 18, 36, 7, 21 |
| 1000 m | 2, 1, 1 | 0 | 0 | 52, 21, 14, 20 |

- **100 m is too tight.** It gives the most fragmented tracks and false "over it now" alerts while the drone
  is still far out. The rays from clusters 100 m apart meet at a shallow angle, so range is poorly fixed.
- **Wider spacing** puts the corner clusters further out, so the drone is detected a little earlier (mid 900 →
  1,190 m). It also gives fewer track fragments and no false take-cover in this sample. The cost is that firm,
  correctly rated tracks form later for the low, fast FPV.
- **300–600 m is the best compromise here.** 300 m gave the best real warning for all three classes, and 600 m
  gave no false alerts. This updates design 13.5 ("100 vs 150 m: no difference"), which was measured before the
  directions were steered by the comb.
- **The loud two-stroke** is heard across the whole 3 km. Its track error is 50–300 m at long range, because a
  bearing error of a few degrees times 2–3 km is about 100 m. That is expected, and it shrinks as it closes.

## Recommendation (needs a decision from the owner)

1. **Spacing:** about 300–600 m between clusters for the site box (instead of 100 m). Then re-run this study
   with more seeds before buying hardware.
2. **The bigger lever is an early alert tier.** It would be an "approaching" warning, sent when the CNN detection
   and the bearings from two or more clusters point at the site, before the track is firm or rated. That
   could add 1–3 minutes of warning here, at the cost of more false warnings. It changes who gets alerted and
   when (D-34, D-35, design 13.6), so it is the owner's call and is **not built**. Outer tripwire clusters
   (design Q46) are the other way to buy time.
