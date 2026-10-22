"""Figures for results/real_world/REAL_TESTS.md from the real-audio result tables.

    python -m analysis.figures --out results/real_world
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERVO = {f"BACKGROUND_{i:03d}.wav" for i in (1, 6, 11, 13, 21, 22, 26)}
plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def any_drone(df, m, th=0.5):
    return (df[f"{m}_drone_electric"] >= th) | (df[f"{m}_drone_engine"] >= th)


def idmt_range(src, out):
    df = pd.read_pickle(src)
    edges = [0, 25, 50, 75, 100, 125, 150, 200, 250]
    df["rb"] = pd.cut(df.r, edges)
    df["det"] = any_drone(df, "B")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    mids = np.array([(a + b) / 2 for a, b in zip(edges[:-1], edges[1:], strict=True)])
    for color, (drone, g) in zip(SERIES, df.groupby("drone"), strict=False):
        s = g.groupby("rb", observed=False).det.agg(["mean", "size"])
        ok = s["size"].to_numpy() >= 250                     # >= 10 s of audio per bin
        x, y = mids[ok], 100 * s["mean"].to_numpy()[ok]
        ax.plot(x, y, color=color, lw=2, marker="o", ms=8, mec=SURFACE, mew=2)
        ax.annotate(drone, (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=11)
    ax.set_xlabel("horizontal distance to drone (m)")
    ax.set_ylabel("frames detected (%)")
    ax.set_ylim(0, 105)
    ax.set_xlim(0, 270)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_title("Model B on real flights (Fraunhofer IDMT): detection vs distance", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(Path(out) / "idmt_detection_vs_distance.png", dpi=120)


def svanstrom_bars(src, out):
    df = pd.read_pickle(src)
    df["group"] = np.where(df.file.isin(SERVO), "background\n(servo motor)",
                           np.where(df.category == "background", "background\n(no motor)", df.category))
    order = ["drone", "helicopter", "background\n(servo motor)", "background\n(no motor)"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.26
    for k, (m, color) in enumerate(zip("BAC", SERIES, strict=True)):
        v = [100 * np.mean([((p[:, 0] >= 0.5) | (p[:, 1] >= 0.5)).mean() for p in df[df.group == g][f"{m}_p"]]) for g in order]
        xs = np.arange(len(order)) + (k - 1) * (w + 0.02)
        ax.bar(xs, v, w, color=color, label={"B": "B (CNN14, log-freq)", "A": "A (CNN14, mel)", "C": "C (small, log-freq)"}[m])
        for x, y in zip(xs, v, strict=True):
            ax.text(x, y + 1.5, f"{y:.0f}", ha="center", fontsize=9, color=INK2)
    ax.set_xticks(np.arange(len(order)), order)
    ax.set_ylabel("frames flagged as drone (%)")
    ax.set_ylim(0, 122)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=10)
    ax.set_title("Real clips (Svanström et al.): drone should be high, the rest low", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(Path(out) / "svanstrom_flag_rate.png", dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/real_world")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    idmt_range("analysis/out/real_idmt.pkl", a.out)
    svanstrom_bars("analysis/out/real_svanstrom.pkl", a.out)


if __name__ == "__main__":
    main()
