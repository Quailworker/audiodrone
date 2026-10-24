"""Score the frequency tracker (loc/freqtrack.py) against the true rotor rates of the localisation scenes.

    python -m loc.eval_freq --scenes data/loc_ft/flat_ground_100 --out results/freqtrack --workers 4

Per scene and cluster it runs the tracker on the reference mic and writes <scene>.freq.npz next to the
audio (reused by later runs and by loc.evaluate --doppler). Truth per drone, cluster and frame: the mean
rotor shaft rate at emission time, Doppler-shifted by the true radial speed (loc/render.py truth()).

Metrics
- lock: a tone is locked to a drone when f / (m * shaft_rx) is within LOCK_REL of 1 for some rotor and
  harmonic number m in 1..4 or 1-3 x the blade count. Detection = locked frames / frames where the drone is audible (tone-line
  SNR >= 6 dB at that cluster, the CNN label rule).
- pitch error of locked tones, relative to the nearest rotor line (median, %).
- Doppler: for two clusters both locked on the same drone in a frame, rdot_i - rdot_j from the pitch
  ratio (freqtrack.rdot_difference) vs the truth. "eff" truth includes the RPM change between the two
  emission times (what the pitch ratio really measures); "kin" is the pure kinematic rdot_i - rdot_j that
  the Kalman filter's model assumes.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from loc.freqtrack import C, HarmonicTracker, rdot_difference

LOCK_REL, AUD_DB = 0.005, 6.0


def track_scene(stem, force=False):
    out = Path(stem + ".freq.npz")
    if out.exists() and not force:
        return dict(np.load(out))
    meta = json.loads(Path(stem + ".json").read_text())
    tf = np.load(stem + ".truth.npz")["frame_t"]
    ht = HarmonicTracker(meta["fs"])
    res = {}
    for k in range(len(meta["centres"])):
        x, _ = sf.read(f"{stem}.c{k}.flac", dtype="float32")
        r = ht.run(x[:, 0], tf)
        for key, v in r.items():
            res[f"c{k}/{key}"] = v.astype(np.float32) if v.dtype.kind == "f" else v.astype(np.int16)
    np.savez_compressed(out, **res)
    return res


def received_lines(truth, i, k):
    """(rotors, frames) shaft rate heard at cluster k, and the audible mask (frames,)."""
    tau = truth[f"d{i}/c{k}/tau"]
    dop = 1 / (1 + truth[f"d{i}/c{k}/rdot"] / C)
    lines = np.stack([np.interp(tau, truth[f"d{i}/t"], s) for s in truth[f"d{i}/shaft_hz"]]) * dop
    return lines


def lock(f, lines, blades):
    """Harmonic number m (0 = not locked) and relative pitch error, per frame. f (F,), lines (R, F).
    Only family fundamentals count: m in 1..4 (shaft rate and low orders) or 1-3 x the blade rate; a
    fixed tolerance at any m <= 12 would match by chance ~1 % x m of the time."""
    m = np.round(f[None] / lines)
    err = f[None] / (np.maximum(m, 1) * lines) - 1
    ok = np.isin(m, [1, 2, 3, 4, blades, 2 * blades, 3 * blades]) & (np.abs(err) < LOCK_REL)
    best = np.argmin(np.where(ok, np.abs(err), np.inf), 0)
    F = np.arange(len(f))
    hit = ok[best, F]
    return np.where(hit, m[best, F], 0).astype(int), np.where(hit, err[best, F], np.nan)


def score_scene(stem, force=False):
    meta = json.loads(Path(stem + ".json").read_text())
    truth = dict(np.load(stem + ".truth.npz"))
    fr = track_scene(stem, force)
    K = len(meta["centres"])
    cats = [s for s in meta["sources"] if s["category"].startswith("drone_")]
    n_frames = len(truth["frame_t"])
    rows, dop = [], []
    locked = {}                                                     # (i, k) -> (F,) slot index or -1
    any_aud = np.zeros((K, n_frames), bool)
    for i, q in enumerate(cats):
        for k in range(K):
            lines = received_lines(truth, i, k)
            aud = truth.get(f"c{k}/{q['category']}/lines01", np.zeros((n_frames, 2)))[:, 1] >= AUD_DB
            any_aud[k] |= aud
            slot = np.full(n_frames, -1)
            m_all = np.zeros(n_frames, int)
            e_all = np.full(n_frames, np.nan)
            ef_all = np.full(n_frames, np.nan)
            for s in range(fr[f"c{k}/f0"].shape[1]):
                m, e = lock(fr[f"c{k}/f0"][:, s].astype(float), lines, q["blades"])
                _, ef = lock(fr[f"c{k}/f0_fine"][:, s].astype(float), lines, q["blades"])
                new = (m > 0) & (slot < 0)
                slot[new], m_all[new], e_all[new], ef_all[new] = s, m[new], e[new], ef[new]
            locked[(i, k)] = slot
            for f in np.flatnonzero(aud | (slot >= 0)):
                rows.append({"scene": Path(stem).name, "drone": i, "cls": q["cls"], "power": q["power"],
                             "blades": q["blades"], "rotors": q["rotors"], "cluster": k, "frame": int(f),
                             "aud": bool(aud[f]), "locked": bool(slot[f] >= 0), "m": int(m_all[f]),
                             "err": e_all[f], "err_fine": ef_all[f], "r": float(truth[f"d{i}/c{k}/r"][f])})
        for k, j in [(a, b) for a in range(K) for b in range(a + 1, K)]:
            sk, sj = locked[(i, k)], locked[(i, j)]
            for f in np.flatnonzero((sk >= 0) & (sj >= 0)):
                fk = float(fr[f"c{k}/f0_fine"][f, sk[f]])
                fj = float(fr[f"c{j}/f0_fine"][f, sj[f]])
                if not (np.isfinite(fk) and np.isfinite(fj)):
                    continue
                lk = received_lines(truth, i, k)[:, f].mean()
                lj = received_lines(truth, i, j)[:, f].mean()
                dop.append({"scene": Path(stem).name, "cls": q["cls"], "rotors": q["rotors"], "frame": int(f),
                            "meas": rdot_difference(fk, fj), "eff": -C * np.log(lk / lj),
                            "kin": float(truth[f"d{i}/c{k}/rdot"][f] - truth[f"d{i}/c{j}/rdot"][f])})
    # tones where no drone is audible at that cluster (confusers, or drones below the label rule)
    for k in range(K):
        on = np.isfinite(fr[f"c{k}/f0"][:, 0]) & ~any_aud[k]
        rows.append({"scene": Path(stem).name, "cluster": k, "quiet_frames": int((~any_aud[k]).sum()),
                     "quiet_tone_frames": int(on.sum())})
    return pd.DataFrame(rows), pd.DataFrame(dop)


def summary(df, dop):
    d = df[df["aud"].eq(True)]                                              # object column (NaN on summary rows)
    out = {"frames_audible": len(d), "detect": float(d["locked"].mean())}
    lk = df[df["locked"].eq(True)]
    out["pitch_err_pct_coarse"] = float(np.nanmedian(np.abs(lk["err"])) * 100)
    out["pitch_err_pct_fine"] = float(np.nanmedian(np.abs(lk["err_fine"])) * 100)
    out["harmonic_m"] = {int(m): int(n) for m, n in lk["m"].value_counts().sort_index().items()}
    out["by_class"] = {c: {"detect": float(g["locked"].mean()), "frames": len(g)} for c, g in d.groupby("cls")}
    bins = [0, 50, 100, 200, 400, 1e5]
    d = d.assign(rb=pd.cut(d["r"], bins))
    out["detect_by_range_m"] = {str(b): float(g["locked"].mean()) for b, g in d.groupby("rb", observed=True)}
    q = df[df["quiet_frames"].notna()]
    out["tone_rate_no_drone_audible"] = float(q["quiet_tone_frames"].sum() / max(q["quiet_frames"].sum(), 1))
    if len(dop):
        e_eff = (dop["meas"] - dop["eff"]).abs()
        e_kin = (dop["meas"] - dop["kin"]).abs()
        out["doppler_pairs"] = len(dop)
        out["doppler_err_ms_vs_eff"] = {"median": float(e_eff.median()), "p90": float(e_eff.quantile(0.9))}
        out["doppler_err_ms_vs_kin"] = {"median": float(e_kin.median()), "p90": float(e_kin.quantile(0.9))}
        out["doppler_by_class_median_ms_kin"] = {c: float((g["meas"] - g["kin"]).abs().median()) for c, g in dop.groupby("cls")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--out", default="results/freqtrack")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="re-run the tracker even if .freq.npz exists")
    a = ap.parse_args()
    stems = sorted(str(p)[:-5] for p in Path(a.scenes).glob("L*.json") if Path(str(p)[:-5] + ".c0.flac").exists())
    if a.limit:
        stems = stems[: a.limit]
    dfs, dops = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for n, (df, dp) in enumerate(ex.map(partial(score_scene, force=a.force), stems), 1):
            dfs.append(df)
            dops.append(dp)
            if n % 10 == 0:
                print(f"{n}/{len(stems)}", flush=True)
    df, dop = pd.concat(dfs, ignore_index=True), pd.concat(dops, ignore_index=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s = summary(df, dop)
    s["scenes"] = len(stems)
    s["source"] = a.scenes
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    df.to_csv(out / "frames.csv.gz", index=False)
    dop.to_csv(out / "doppler.csv.gz", index=False)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
