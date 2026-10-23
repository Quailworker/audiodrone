"""Tables and figures for results/loc_v1 from loc/evaluate.py and loc/idmt.py outputs.

    python -m loc.report --out results/loc_v1
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from loc.doa import DOA  # noqa: E402
from loc.evaluate import COH_GATE, RANGE_BINS  # noqa: E402
from sim.layouts import cluster_offsets  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LABEL = {"tetra_pole_100": "tetrahedron, 2 m pole", "flat_ground_100": "flat, on the ground",
         "tetra_ground_100": "tetrahedron, on the ground", "tetra_pole_150": "tetrahedron, pole, 150 m field"}
plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)


def _line_by_bins(ax, x_mid, ys, labels, ymax=None, legend=False):
    for color, (lab, y) in zip(SERIES, zip(labels, ys, strict=True), strict=False):
        ok = np.isfinite(y)
        ax.plot(x_mid[ok], y[ok], color=color, lw=2, marker="o", ms=7, mec=SURFACE, mew=2, label=lab)
        if ok.any() and not legend:
            ax.annotate(lab, (x_mid[ok][-1], y[ok][-1]), xytext=(8, 0), textcoords="offset points", va="center",
                        color=INK, fontsize=10)
    if legend:
        ax.legend(frameon=False, fontsize=10)
    if ymax:
        ax.set_ylim(0, ymax)


def fig_direction_vs_range(data, configs, out):
    mids = np.array([12.5, 37.5, 75, 125, 200, 300])
    ys, labs = [], []
    for cfg in configs:
        d = data[cfg]
        g = d[d.coh_bf >= COH_GATE["bf"]]
        y = g.groupby(pd.cut(g.r, RANGE_BINS), observed=False).err_bf.median().to_numpy()
        ys.append(y)
        labs.append(LABEL.get(cfg, cfg))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    _line_by_bins(ax, mids, ys, labs)
    ax.set_xlim(0, 400)
    _style(ax, "Direction error vs range (reflection-aware beamformer, gated)",
           "range to cluster (m); the last point is 250 m and beyond", "median error (deg)")
    fig.tight_layout()
    fig.savefig(out / "direction_error_vs_range.png", dpi=120)
    plt.close(fig)


def fig_methods(summary, configs, out):
    methods = [("srp", "pairwise SRP-PHAT"), ("ls", "pair-delay least squares"), ("bf", "reflection-aware beamformer")]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    w = 0.26
    for i, (m, lab) in enumerate(methods):
        v = [summary[c]["direction"]["gated"].get(m, {}).get("median", np.nan) for c in configs]
        xs = np.arange(len(configs)) + (i - 1) * (w + 0.02)
        ax.bar(xs, v, w, color=SERIES[i], label=lab)
        for x, y in zip(xs, v, strict=True):
            if np.isfinite(y):
                ax.text(x, y + 0.4, f"{y:.1f}", ha="center", fontsize=9, color=INK2)
    ax.set_xticks(np.arange(len(configs)), [LABEL.get(c, c).replace(", ", ",\n") for c in configs])
    ax.set_ylim(0, np.nanmax([summary[c]["direction"]["gated"].get("ls", {}).get("median", 0) for c in configs]) * 1.3)
    ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=10)
    _style(ax, "Median direction error by method (audible frames, gated)", "", "median error (deg)")
    fig.tight_layout()
    fig.savefig(out / "direction_error_by_method.png", dpi=120)
    plt.close(fig)


def fig_variants(data, configs, out):
    """Paired comparison on the scenes that have every experiment: whole band vs 2 kHz vs oracle comb."""
    variants = [("bf", "beamformer 100-860 Hz"), ("bf_band2k", "100-2000 Hz"), ("bf_comb", "comb lines, 100-860 Hz"),
                ("bf_comb2k", "comb lines, 100-2000 Hz")]
    cfgs = [c for c in configs if "err_bf_comb2k" in data[c] and data[c].err_bf_comb2k.notna().any()]
    if not cfgs:
        return {}
    fig, ax = plt.subplots(figsize=(9, 4.8))
    w = 0.2
    table = {}
    for i, (m, lab) in enumerate(variants):
        v = []
        for c in cfgs:
            d = data[c][data[c][[f"err_{x}" for x, _ in variants]].notna().all(1)]
            g = d[d[f"coh_{m}"] >= COH_GATE["bf"]]
            v.append(float(g[f"err_{m}"].median()))
            table.setdefault(c, {"frames": int(len(d)), "scenes": int(d.scene.nunique())})[m] = {
                "median": v[-1], "gross_gt10": float((g[f"err_{m}"] > 10).mean()), "kept": float(len(g) / max(len(d), 1))}
        xs = np.arange(len(cfgs)) + (i - 1.5) * (w + 0.02)
        ax.bar(xs, v, w, color=SERIES[i], label=lab)
        for x, y in zip(xs, v, strict=True):
            ax.text(x, y + 0.15, f"{y:.1f}", ha="center", fontsize=9, color=INK2)
    ax.set_xticks(np.arange(len(cfgs)), [LABEL.get(c, c) for c in cfgs])
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    _style(ax, "Direction finding on the drone's own lines (oracle comb), same frames", "", "median error (deg)")
    fig.tight_layout()
    fig.savefig(out / "direction_error_comb_vs_band.png", dpi=120)
    plt.close(fig)
    return table


def fig_height(tracks, configs, out):
    bins = [0, 25, 50, 75, 100, 150, 250]
    mids = np.array([12.5, 37.5, 62.5, 87.5, 125, 200])
    ys, labs = [], []
    for cfg in configs:
        t = tracks[cfg]
        t = t[(t.n_drones == 1) & t.pos_err.notna()]
        ys.append(t.h_err.abs().groupby(pd.cut(t.rho_min, bins), observed=False).median().to_numpy())
        labs.append(LABEL.get(cfg, cfg))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    _line_by_bins(ax, mids, ys, labs)
    ax.set_xlim(0, 260)
    _style(ax, "Tracked height error vs distance to the nearest cluster (single drone)",
           "horizontal distance to nearest cluster (m)", "median |height error| (m)")
    fig.tight_layout()
    fig.savefig(out / "height_error_vs_distance.png", dpi=120)
    plt.close(fig)


def fig_example(tracks, data_dir, cfg, out):
    t = tracks[cfg]
    t = t[t.n_drones == 1]
    tracked = t[t.pos_err.notna()]
    n = tracked.groupby("scene").size()
    med = tracked.groupby("scene").pos_err.median()[n >= 200]     # tracked for at least 20 s
    if med.empty:
        return
    scene = (med - tracked.pos_err.median()).abs().idxmin()     # typical error, not the best scene
    tr = np.load(Path(data_dir) / cfg / f"{scene}.truth.npz")
    meta = json.loads((Path(data_dir) / cfg / f"{scene}.json").read_text())
    g = t[t.scene == scene]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
    p = tr["d0/p"]
    tt = tr["d0/t"]
    sel = (tt >= 0) & (tt <= g.t.max())
    a1.plot(p[sel, 0], p[sel, 1], color=INK2, lw=2, label="true path")
    c = np.array(meta["centres"])
    a1.scatter(c[:, 0], c[:, 1], marker="^", s=70, color=SERIES[1], label="clusters", zorder=3)
    est = g[g.pos_err.notna()]
    a1.scatter(est.est_x, est.est_y, s=8, color=SERIES[0], label="track", zorder=2)
    a1.set_aspect("equal", "datalim")
    a1.legend(frameon=False, fontsize=9, loc="best")
    _style(a1, f"{scene} ({meta['sources'][0]['cls']}): top view", "x (m)", "y (m)")
    a2.plot(g.t, g.h, color=INK2, lw=2, label="true height")
    a2.scatter(est.t, est.est_z, s=6, color=SERIES[0], label="tracked height")
    a2.legend(frameon=False, fontsize=9)
    _style(a2, "height vs time", "time (s)", "height (m)")
    fig.tight_layout()
    fig.savefig(out / "example_track.png", dpi=120)
    plt.close(fig)


def fig_idmt(idmt_csv, out):
    df = pd.read_csv(idmt_csv)
    bins = [0, 25, 50, 100, 150, 200, 450]
    mids = np.array([12.5, 37.5, 75, 125, 175, 300])
    ys, labs = [], []
    for m, lab in (("bf_free", "beamformer"), ("srp", "pairwise SRP-PHAT"), ("ls", "pair-delay LS")):
        ys.append(df.groupby(pd.cut(df.rho, bins), observed=False)[f"{m}_err_cal"].median().to_numpy())
        labs.append(lab)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    _line_by_bins(ax, mids, ys, labs, legend=True)
    ax.set_xlim(0, 330)
    _style(ax, "Real IDMT flights: azimuth error vs GPS (heading-calibrated, leave one flight out)",
           "horizontal distance (m)", "median azimuth error (deg)")
    fig.tight_layout()
    fig.savefig(out / "idmt_azimuth_error_vs_range.png", dpi=120)
    plt.close(fig)


def ops_table():
    rows = {}
    for fs in (8000, 4000):
        d = DOA(cluster_offsets("tetra"), fs, device="cpu")
        o = d.ops_per_frame()
        rows[f"{fs} Hz"] = {k: (int(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in o.items()}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/loc_v1")
    ap.add_argument("--data", default="data/loc_v1")
    a = ap.parse_args()
    out = Path(a.out)
    summary = json.loads((out / "summary.json").read_text())
    configs = [c for c in LABEL if c in summary]
    ev = Path(a.data) / "eval"
    data = {c: pd.read_pickle(ev / f"{c}_direction_frames.pkl") for c in configs}
    tracks = {c: pd.read_pickle(ev / f"{c}_track_bf.pkl") for c in configs}
    fig_direction_vs_range(data, configs, out)
    fig_methods(summary, configs, out)
    (out / "comb_vs_band.json").write_text(json.dumps(fig_variants(data, configs, out), indent=1))
    fig_height(tracks, configs, out)
    fig_example(tracks, a.data, configs[0], out)
    if (out / "idmt" / "idmt_frames.csv.gz").exists():
        fig_idmt(out / "idmt" / "idmt_frames.csv.gz", out)
    (out / "ops_per_frame.json").write_text(json.dumps(ops_table(), indent=1))
    print("figures and ops table written to", out)


if __name__ == "__main__":
    main()
