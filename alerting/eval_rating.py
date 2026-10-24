"""Does the D-32 rating separate drone tracks from false tracks? (open questions 28, 41)

    python -m alerting.eval_rating --config flat_ground_100 --method bf --out results/system_demo/rating_eval.json

Runs the tracker on every scene, rates every confirmed track each frame (alerting/rating.py; oracle
detection as in Phase 1, so p_cnn is the same for all tracks and only height, speed and persistence
separate them), and labels each track-frame "drone" (within 50 m of a true drone) or "false" (on
something else). Reports how track time splits over the rating levels.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from alerting.rating import Evidence, rate
from alerting.replay import P_CNN, scene_inputs
from loc.track import Track, Tracker
from sim.acoustics import interp_path

MATCH_M = 50.0


def scene_rows(args):
    stem, method = args
    try:
        meta, truth, tf, centres, aud, meas = scene_inputs(stem, method)
    except SystemExit:
        return []
    n_d = sum(s["category"].startswith("drone") for s in meta["sources"])
    P = [interp_path(truth[f"d{i}/t"], truth[f"d{i}/p"], tf) for i in range(n_d)]
    Track._next = 0
    trk = Tracker(centres, flat=meta["cluster"] == "flat", sigma_u=0.04)
    rows, heard_since = [], None
    for i, t in enumerate(tf):
        tracks = trk.step(t, meas[i])
        heard = bool(aud[:, i].any())
        heard_since = (heard_since if heard_since is not None else t) if heard else None
        det_s = 0.0 if heard_since is None else t - heard_since + 0.1
        for tr in tracks:
            if tr.n_upd < 5:
                continue
            dist = min(np.linalg.norm(tr.s[:3] - P[k][i]) for k in range(n_d))
            r = rate(Evidence(P_CNN, det_s, 1.0, float(tr.s[2]), float(np.sqrt(tr.P[2, 2])),
                              float(np.linalg.norm(tr.s[3:6])), int(aud[:, i].sum())))
            rows.append(("drone" if dist <= MATCH_M else "false", r.level, float(tr.s[2]), float(np.linalg.norm(tr.s[3:6]))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="flat_ground_100")
    ap.add_argument("--method", default="bf")
    ap.add_argument("--out", default="results/system_demo/rating_eval.json")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    stems = sorted(str(p)[:-8] for p in Path("data/loc_v1", a.config).glob("L*.doa.npz"))
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for rs in ex.map(scene_rows, [(s, a.method) for s in stems]) for r in rs]
    out = {"config": a.config, "method": a.method, "scenes": len(stems), "track_frames": len(rows)}
    for lab in ("drone", "false"):
        sel = [r for r in rows if r[0] == lab]
        out[lab] = {"frames": len(sel), **{lev: round(sum(r[1] == lev for r in sel) / max(len(sel), 1), 3)
                                          for lev in ("likely", "possible", "ground")},
                    "height_median_m": round(float(np.median([r[2] for r in sel])), 1) if sel else None,
                    "speed_median_m_s": round(float(np.median([r[3] for r in sel])), 1) if sel else None}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    prev[f"{a.config}/{a.method}"] = out
    Path(a.out).write_text(json.dumps(prev, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
