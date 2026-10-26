"""Ghost tracks: re-run only the tracker on saved runs' direction measurements and count tracks against drones.

    python scripts/retrack_eval.py runs_terra/<id> [runs_border/<id> ...]

Each run directory holds the scenario and the full result (cluster positions, the directions used each frame,
the simulated drone paths). The tracker is run as the pipeline runs it (terrain as the ground), and per frame:
  tracks    confirmed tracks (at least 5 updates), what an operator sees
  held      drones with a confirmed track within MATCH_M (horizontal)
  extra     tracks beyond one per held drone: ghosts and duplicates
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loc.track import Track, Tracker
from terra.scenario import load_terrain

MATCH_M = 150.0


def retrack(run_dir, **kw):
    run_dir = Path(run_dir)
    scn = json.loads((run_dir / "scenario.json").read_text())
    res = json.loads((run_dir / "result.json").read_text())
    terrain = load_terrain(scn)
    centres = np.array([[c["x"], c["y"], c["z"]] for c in res["clusters"]])
    used = np.asarray(res["used"], bool)
    U = np.asarray(res["u"], float)
    tf = np.asarray(res["t"])
    truth = [(np.asarray(d["t"]), np.asarray(d["p"])) for d in res["drones"]]
    Track._next = 0
    trk = Tracker(centres, flat=True, sigma_u=0.04, ground=terrain.height, birth_range=4000.0, **kw)
    n_tracks, n_held, n_extra, ids = [], [], [], set()
    for i, t in enumerate(tf):
        meas = [(k, U[k, i]) for k in range(len(centres)) if used[k, i]]
        tracks = [tr for tr in trk.step(t, meas) if tr.n_upd >= 5]
        ids |= {tr.id for tr in tracks}
        pos = [np.array([np.interp(t, tt, pp[:, 0]), np.interp(t, tt, pp[:, 1])]) for tt, pp in truth]
        held = sum(1 for p in pos if tracks and min(np.hypot(*(tr.s[:2] - p)) for tr in tracks) <= MATCH_M)
        n_tracks.append(len(tracks))
        n_held.append(held)
        n_extra.append(max(0, len(tracks) - held))
    n_tracks, n_held, n_extra = map(np.asarray, (n_tracks, n_held, n_extra))
    any_track = n_tracks > 0
    return {"run": run_dir.name, "drones": len(truth), "track_ids": len(ids), "max_tracks": int(n_tracks.max()),
            "frames_with_extra": round(float((n_extra > 0).mean()), 3),
            "mean_extra_when_tracking": round(float(n_extra[any_track].mean()), 2) if any_track.any() else 0.0,
            "drone_frames_held": round(float(n_held.sum() / (len(truth) * len(tf))), 3)}


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(json.dumps(retrack(d)))
