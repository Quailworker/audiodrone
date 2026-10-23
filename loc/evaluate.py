"""Score direction finding and tracking against ground truth (tasks/phase1_localisation.md step 4).

    python -m loc.evaluate --configs tetra_pole_100 flat_ground_100 [...] --out results/loc_v1

Frames count only where the drone is audible at that cluster by the CNN label rule (tone-line SNR,
loc/render.py), i.e. where a detector would plausibly have fired. Direction statistics use single-drone
scenes; tracking is reported for single- and two-drone scenes separately.

Tracking uses "oracle detection": a cluster contributes a direction in a frame when the drone is audible
there by the label rule and the direction's coherence passes the method's gate. That isolates
localisation from detection errors (the CNN's own recall is ~92 % in sim, design 5.9).
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from loc.doa import angle_deg
from loc.track import run_tracker
from sim.acoustics import interp_path

RANGE_BINS = [0, 25, 50, 100, 150, 250, 5000]
EL_BINS = [-90, 5, 10, 20, 40, 60, 90]
MATCH_M = 50.0                    # a track within this distance of a drone counts as tracking it
COH_GATE = {"bf": 0.5, "srp": 0.3, "ls": 0.3}


def load_scene(stem):
    meta = json.loads(Path(stem + ".json").read_text())
    truth = dict(np.load(stem + ".truth.npz"))
    doa_path = Path(stem + ".doa.npz")
    doa = dict(np.load(doa_path)) if doa_path.exists() else None
    return meta, truth, doa


def drone_cats(meta):
    return [s["category"] for s in meta["sources"] if s["category"].startswith("drone_")]


def audible(truth, meta, k, db=6.0):
    """Per frame: is any drone audible at cluster k (tone-line rule)?"""
    a = np.zeros(len(truth["frame_t"]), bool)
    for cat in set(drone_cats(meta)):
        key = f"c{k}/{cat}/lines01"
        if key in truth:
            a |= truth[key][:, 1] >= db
    return a


# --- direction finding ---------------------------------------------------------------------

def direction_rows(stem):
    meta, truth, doa = load_scene(stem)
    if doa is None or len(drone_cats(meta)) != 1:
        return None
    q = meta["sources"][0]
    cat = q["category"]
    methods = sorted({k.split("/")[0] for k in doa})
    rows = []
    for k in range(len(meta["centres"])):
        aud = audible(truth, meta, k)
        if not aud.any():
            continue
        u = truth[f"d0/c{k}/u"][aud]
        d = {"scene": Path(stem).name, "config": meta["config"], "k": k, "frame": np.flatnonzero(aud),
             "r": truth[f"d0/c{k}/r"][aud], "rho": truth[f"d0/c{k}/rho"][aud],
             "el": np.degrees(np.arcsin(np.clip(u[:, 2], -1, 1))),
             "snr_df860": truth[f"c{k}/{cat}/snr_df860"][aud], "snr_df2k": truth[f"c{k}/{cat}/snr_df2k"][aud],
             "cls": q["cls"], "bed_level": meta["bed_level"], "family": q.get("family", "")}
        for m in methods:
            ue = doa[f"{m}/c{k}/u"][aud].astype(float)
            d[f"err_{m}"] = angle_deg(ue, u).astype(np.float32)
            d[f"elerr_{m}"] = (np.degrees(np.arcsin(np.clip(ue[:, 2], -1, 1))) - d["el"]).astype(np.float32)
            d[f"coh_{m}"] = doa[f"{m}/c{k}/coh"][aud].astype(np.float32)
        rows.append(pd.DataFrame(d))
    return pd.concat(rows) if rows else None


def stats(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"n": 0}
    return {"n": int(len(v)), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "gross_gt10": float(np.mean(v > 10))}


def direction_summary(df, methods):
    out = {"frames": int(len(df)), "scenes": int(df.scene.nunique()), "overall": {}, "by_range": {}, "by_el": {},
           "by_cls": {}, "by_bed": {}, "by_snr": {}, "gated": {}, "near_horizon_el": {}}
    rb = pd.cut(df.r, RANGE_BINS).astype(str)
    eb = pd.cut(df.el, EL_BINS).astype(str)
    sb = pd.cut(df.snr_df860, [-100, -10, -5, 0, 5, 10, 100]).astype(str)
    bb = pd.cut(df.bed_level, [30, 45, 55, 65, 75]).astype(str)
    for m in methods:
        e = df[f"err_{m}"]
        if e.isna().all():
            continue
        out["overall"][m] = stats(e)
        out["by_range"][m] = {k: stats(v) for k, v in e.groupby(rb)}
        out["by_el"][m] = {k: stats(v) for k, v in e.groupby(eb)}
        out["by_cls"][m] = {k: stats(v) for k, v in e.groupby(df.cls)}
        out["by_bed"][m] = {k: stats(v) for k, v in e.groupby(bb)}
        out["by_snr"][m] = {k: stats(v) for k, v in e.groupby(sb)}
        valid = e.notna()
        g = (df[f"coh_{m}"] >= COH_GATE.get(m.split("_")[0], 0.2)) & valid
        out["gated"][m] = {"gate": COH_GATE.get(m.split("_")[0], 0.2), "kept": float(g[valid].mean()), **stats(e[g])}
        if m != "bf" and "err_bf" in df and valid.sum() < len(df):
            # variants exist on a subset of scenes: compare with bf on exactly the same frames
            gb = (df.coh_bf >= COH_GATE["bf"]) & valid
            out.setdefault("paired_vs_bf", {})[m] = {"frames": int(valid.sum()), m: stats(e[g]), "bf": stats(df.err_bf[gb])}
        low = (df.el < 15) & g
        el_err = df.loc[low, f"elerr_{m}"].abs()
        out["near_horizon_el"][m] = {"frames": int(low.sum()), "median_abs_el_err": float(el_err.median()) if len(el_err) else None,
                                     "p90_abs_el_err": float(el_err.quantile(0.9)) if len(el_err) else None}
    return out


# --- tracking -----------------------------------------------------------------------------

def track_scene(args):
    stem, method, sigma_u = args
    meta, truth, doa = load_scene(stem)
    if doa is None or f"{method}/c0/u" not in doa:
        return None
    tf = truth["frame_t"]
    centres = np.asarray(meta["centres"])
    K = len(centres)
    gate = COH_GATE.get(method.split("_")[0], 0.2)
    aud = np.stack([audible(truth, meta, k) for k in range(K)])                  # (K, F)
    ok = aud & np.stack([doa[f"{method}/c{k}/coh"].astype(float) >= gate for k in range(K)])
    meas = [[(k, doa[f"{method}/c{k}/u"][i].astype(float)) for k in range(K) if ok[k, i]] for i in range(len(tf))]
    res = run_tracker(tf, meas, centres, flat=meta["cluster"] == "flat", sigma_u=sigma_u)
    n_d = len(drone_cats(meta))
    P = [interp_path(truth[f"d{i}/t"], truth[f"d{i}/p"], tf) for i in range(n_d)]
    V = [interp_path(truth[f"d{i}/t"], truth[f"d{i}/v"], tf) for i in range(n_d)]
    rows = []
    false_frames, track_frames = 0, 0
    for f, tracks in enumerate(res):
        confirmed = [tr for tr in tracks if tr[3] >= 5]
        track_frames += len(confirmed)
        for tr in confirmed:
            if min(np.linalg.norm(tr[1][:3] - P[i][f]) for i in range(n_d)) > MATCH_M:
                false_frames += 1
        for i in range(n_d):
            rho_c = np.hypot(*(P[i][f][:2] - centres[:, :2]).T)
            row = {"scene": Path(stem).name, "drone": i, "frame": f, "t": tf[f], "n_aud": int(aud[:, f].sum()),
                   "n_meas": int(ok[:, f].sum()), "rho_min": float(rho_c.min()), "h": float(P[i][f][2]),
                   "speed": float(np.linalg.norm(V[i][f])), "n_drones": n_d}
            if confirmed:
                d = [np.linalg.norm(tr[1][:3] - P[i][f]) for tr in confirmed]
                j = int(np.argmin(d))
                if d[j] <= MATCH_M:
                    s = confirmed[j][1]
                    row.update(est_x=float(s[0]), est_y=float(s[1]), est_z=float(s[2]),
                               pos_err=float(d[j]), h_err=float(s[2] - P[i][f][2]),
                               horiz_err=float(np.hypot(*(s[:2] - P[i][f][:2]))),
                               vel_err=float(np.linalg.norm(s[3:] - V[i][f])), speed_est=float(np.linalg.norm(s[3:])),
                               sd_pos=float(np.sqrt(confirmed[j][2][:3].sum())))
            rows.append(row)
    df = pd.DataFrame(rows)
    df["false_track_frames"] = false_frames
    df["track_frames"] = track_frames
    return df


def track_summary(df):
    s = df[df.n_drones == 1]
    out = {}
    for name, d in (("single_drone", s), ("two_drones", df[df.n_drones == 2])):
        if d.empty:
            continue
        trackable = d.n_aud >= 2
        hit = d.pos_err.notna()
        res = {"frames_audible_2plus": int(trackable.sum()),
               "coverage_audible_2plus": float(hit[trackable].mean()) if trackable.any() else None,
               "coverage_audible_1": float(hit[d.n_aud == 1].mean()) if (d.n_aud == 1).any() else None,
               "pos": stats(d.pos_err[hit]), "height_abs": stats(d.h_err[hit].abs()),
               "horizontal": stats(d.horiz_err[hit]), "velocity": stats(d.vel_err[hit]),
               "rms": {"pos": float(np.sqrt(np.mean(d.pos_err[hit] ** 2))) if hit.any() else None,
                       "height": float(np.sqrt(np.mean(d.h_err[hit] ** 2))) if hit.any() else None,
                       "velocity": float(np.sqrt(np.mean(d.vel_err[hit] ** 2))) if hit.any() else None}}
        rb = pd.cut(d.rho_min, [0, 25, 50, 75, 100, 150, 250, 5000]).astype(str)
        res["height_abs_by_rho_min"] = {k: stats(v.abs()) for k, v in d.h_err[hit].groupby(rb[hit])}
        res["pos_by_rho_min"] = {k: stats(v) for k, v in d.pos_err[hit].groupby(rb[hit])}
        res["by_n_aud"] = {str(k): {"coverage": float(hit[d.n_aud == k].mean()), **stats(d.pos_err[hit & (d.n_aud == k)])}
                           for k in sorted(d.n_aud.unique()) if k > 0}
        # latency: first frame audible at >= 2 clusters -> first frame tracked, per scene and drone
        lat = []
        for _, g in d.groupby(["scene", "drone"]):
            a = g.frame[g.n_aud >= 2]
            if a.empty:
                continue
            tr = g.frame[(g.frame >= a.min()) & g.pos_err.notna()]
            lat.append((tr.min() - a.min()) / 10 if not tr.empty else np.inf)
        lat = np.array(lat)
        res["latency_s"] = {"scenes": int(len(lat)), "never": float(np.mean(~np.isfinite(lat))) if len(lat) else None,
                            "median": float(np.median(lat[np.isfinite(lat)])) if np.isfinite(lat).any() else None}
        per_scene = d.groupby("scene")[["false_track_frames", "track_frames"]].first()
        res["false_track_frac"] = float(per_scene.false_track_frames.sum() / max(per_scene.track_frames.sum(), 1))
        out[name] = res
    return out


def sigma_from(dirdf, method):
    """Per-axis direction noise from the gated median angle error (Rayleigh: median = 1.177 sigma)."""
    if f"err_{method}" not in dirdf or dirdf[f"err_{method}"].isna().all():
        return None
    g = dirdf[f"coh_{method}"] >= COH_GATE.get(method.split("_")[0], 0.2)
    return float(np.radians(dirdf.loc[g, f"err_{method}"].median()) / 1.177)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--data", default="data/loc_v1")
    ap.add_argument("--out", default="results/loc_v1")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--track-methods", nargs="+", default=["bf", "srp", "ls", "bf_band2k", "bf_comb", "bf_comb2k"])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for cfg in a.configs:
        stems = sorted(str(p)[:-8] for p in (Path(a.data) / cfg).glob("L*.doa.npz"))
        with ProcessPoolExecutor(a.workers) as ex:
            dirdf = pd.concat([d for d in ex.map(direction_rows, stems) if d is not None], ignore_index=True)
        methods = [c[4:] for c in dirdf.columns if c.startswith("err_")]
        frames_dir = Path(a.data) / "eval"
        frames_dir.mkdir(parents=True, exist_ok=True)
        dirdf.to_pickle(frames_dir / f"{cfg}_direction_frames.pkl")             # large: kept out of results/
        ds = direction_summary(dirdf, methods)
        summary[cfg] = {"direction": ds, "tracking": {}}
        tracks = {}
        for m in a.track_methods:
            sig = sigma_from(dirdf, m)
            if sig is None:
                continue
            with ProcessPoolExecutor(a.workers) as ex:
                tdf = pd.concat([d for d in ex.map(track_scene, [(s, m, sig) for s in stems]) if d is not None], ignore_index=True)
            tdf.to_pickle(frames_dir / f"{cfg}_track_{m}.pkl")
            tracks[m] = tdf
            summary[cfg]["tracking"][m] = {"sigma_u": sig, "scenes": int(tdf.scene.nunique()), **track_summary(tdf)}
            if m != "bf" and "bf" in tracks and tdf.scene.nunique() < tracks["bf"].scene.nunique():
                same = tracks["bf"][tracks["bf"].scene.isin(tdf.scene.unique())]
                summary[cfg]["tracking"][m]["bf_same_scenes"] = track_summary(same)
        print(cfg, json.dumps({m: ds["gated"].get(m) for m in methods}, indent=None), flush=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
