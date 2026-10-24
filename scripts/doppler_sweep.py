"""Doppler-in-the-EKF noise sweep on the single-drone flat_ground_100 scenes (results/chain_v2/README.md).

    python scripts/doppler_sweep.py        # needs the chain v2 outputs (.doa.npz with bf_trk, .freq.npz)

sigma_lnf = 0 means no Doppler (directions only). Tracking uses the bf_trk directions with the truth
audibility trigger, as in the evaluate.py "bf_trk" / "bf_trk:dop" rows.
"""
import functools
import glob
import json
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import loc.track as T
from loc import evaluate as E

SIGMA_U = 0.0669868141412735          # bf_trk direction sigma, from results/chain_v2/summary.json
GRID = [(0, 1e-3), (0.003, 1e-3), (0.01, 1e-3), (0.03, 1e-3), (0.01, 1e-2), (0.03, 1e-2)]   # (sigma_lnf, q_f)


def one(args):
    stem, sig, qf = args
    meta = json.loads(Path(stem + ".json").read_text())
    cats = [s for s in meta["sources"] if s["category"].startswith("drone_")]
    if len(cats) != 1:
        return None
    E.run_tracker = functools.partial(T.run_tracker, sigma_lnf=sig, q_f=qf)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = E.track_scene((stem, "bf_trk:dop" if sig else "bf_trk", SIGMA_U))
    return None if df is None else df.assign(sig=sig, qf=qf, cls=cats[0]["cls"])


if __name__ == "__main__":
    stems = sorted(p[:-5] for p in glob.glob("data/loc_ft/flat_ground_100/L*.json"))
    with Pool(3) as p:
        dfs = [d for d in p.map(one, [(s, g, q) for g, q in GRID for s in stems]) if d is not None]
    t = pd.concat(dfs)
    t = t[t["n_aud"] >= 2]
    print(t.groupby(["sig", "qf"]).agg(cov=("pos_err", lambda s: s.notna().mean()), horiz=("horiz_err", "median"),
          vel=("vel_err", "median"), vel_p90=("vel_err", lambda s: s.quantile(0.9))).round(2).to_string())
    print(t.groupby(["cls", "sig", "qf"])["vel_err"].median().unstack([1, 2]).round(2).to_string())
