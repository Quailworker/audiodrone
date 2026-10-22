"""Slice the per-frame table (analysis/frames.py) to find where the model fails.

    python -m analysis.slices analysis/out/B_v1.frames.pkl > analysis/out/B_v1.slices.md

Recall is measured only on frames labelled positive (the drone is audible by the label rule), the false
positive rate only on frames labelled negative; ignored frames are reported separately.
"""
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
CLASSES = ["drone_electric", "drone_engine", "ground_engine"]
MIN_FRAMES = 200


def table(df, c, by, kind):
    """Group df by `by`; kind 'recall' (positive frames) or 'fp' (negative frames) for class c."""
    pos = (df[f"y_{c}"] > 0) & (df[f"m_{c}"] > 0)
    neg = (df[f"y_{c}"] == 0) & (df[f"m_{c}"] > 0)
    det = df[f"p_{c}"] >= 0.5
    sel = pos if kind == "recall" else neg
    d = pd.DataFrame({"g": by[sel] if isinstance(by, pd.Series) else df.loc[sel, by], "det": det[sel], "scene": df.scene[sel]})
    g = d.groupby("g", observed=True)
    out = pd.DataFrame({"frames": g.size(), "scenes": g.scene.nunique(), kind: g.det.mean()})
    return out[out.frames >= MIN_FRAMES]


def show(title, t, kind):
    print(f"\n#### {title}\n")
    if t.empty:
        print("(too few frames)")
        return
    t = t.copy()
    t[kind] = (100 * t[kind]).map(lambda v: f"{v:.1f} %" if kind == "recall" else f"{v:.2f} %")
    print(t.to_markdown())


def main():
    df = pd.read_pickle(sys.argv[1])
    df["mic"] = np.where(df.mic_h < 0.5, "surface", "pole 1-3 m")
    df["position"] = pd.cut(np.minimum(df.t, df.t_end), [0, 2, 5, 10, 1e9], labels=["<2 s from edge", "2-5 s", "5-10 s", ">10 s"])
    df["wind"] = pd.cut(df.wind_level, [-1, 1, 50, 60, 70], labels=["none", "30-50", "50-60", "60-70"])
    df["bed_lvl"] = pd.cut(df.bed_level, [30, 45, 55, 65, 75])
    for c in CLASSES[:2]:
        d = df[df.d_category == c]
        if d.empty:
            continue
        print(f"\n## {c}: recall on frames labelled audible\n")
        pos = (d[f"y_{c}"] > 0) & (d[f"m_{c}"] > 0)
        print(f"overall recall {100 * (d[f'p_{c}'][pos] >= 0.5).mean():.1f} % over {pos.sum()} frames")
        show("label margin: 2nd tone region above everything else (dB)", table(d, c, pd.cut(d[f"l1_{c}"], [6, 8, 10, 15, 20, 30, 100]), "recall"), "recall")
        show("drone class", table(d, c, "d_cls", "recall"), "recall")
        show("distance (m)", table(d, c, pd.cut(d.r, [0, 10, 25, 50, 100, 200, 400, 800, 5000]), "recall"), "recall")
        show("elevation (deg)", table(d, c, pd.cut(d.elev, [-90, 5, 15, 30, 50, 70, 80, 85, 90]), "recall"), "recall")
        show("radial speed (m/s, negative = approaching)", table(d, c, pd.cut(d.rdot, [-100, -20, -10, -3, 3, 10, 20, 100]), "recall"), "recall")
        show("received blade-pass frequency (Hz)", table(d, c, pd.cut(d.bpf_rx, [0, 50, 100, 200, 300, 500, 800, 1200, 3000]), "recall"), "recall")
        show("blades per rotor", table(d, c, "d_blades", "recall"), "recall")
        show("rotors", table(d, c, "d_rotors", "recall"), "recall")
        show("flight pattern", table(d, c, "d_family", "recall"), "recall")
        show("number of drones in scene", table(d, c, "n_drones", "recall"), "recall")
        show("background type", table(d, c, "beds", "recall"), "recall")
        show("background level (dBA)", table(d, c, "bed_lvl", "recall"), "recall")
        show("wind on mic (dBA)", table(d, c, "wind", "recall"), "recall")
        show("mic height", table(d, c, "mic", "recall"), "recall")
        show("loudest other source at that moment", table(d, c, "interferer", "recall"), "recall")
        rel = d.interferer_db - d[f"lvl_{c}"]
        show("loudest other source minus drone, in band (dB)", table(d, c, pd.cut(rel, [-100, -20, -10, 0, 10, 20, 100]), "recall"), "recall")
        show("time from clip edge", table(d, c, "position", "recall"), "recall")
        show("held-out conditions", table(d, c, "split", "recall"), "recall")

    for c in CLASSES:
        print(f"\n## {c}: false positive rate on frames labelled negative\n")
        neg = (df[f"y_{c}"] == 0) & (df[f"m_{c}"] > 0)
        print(f"overall FP rate {100 * (df[f'p_{c}'][neg] >= 0.5).mean():.2f} % over {neg.sum()} frames")
        show("loudest non-drone source at that moment", table(df, c, "interferer", "fp"), "fp")
        show("confusers in the scene", table(df.assign(cf=df.confusers.astype(str)), c, "cf", "fp").sort_values("fp", ascending=False).head(25), "fp")
        show("drone present in scene (other class)", table(df, c, df.d_category.astype(str).fillna("none"), "fp"), "fp")
        show("background type", table(df, c, "beds", "fp"), "fp")
        show("wind on mic (dBA)", table(df, c, "wind", "fp"), "fp")
        show("held-out conditions", table(df, c, "split", "fp"), "fp")

    print("\n## Ignored frames (between the label thresholds): how often the model fires\n")
    for c in CLASSES[:2]:
        d = df[df.d_category == c]
        ign = d[f"m_{c}"] == 0
        show(f"{c}, ignored frames by distance", table(d.assign(**{f"y_{c}": 1.0, f"m_{c}": 1.0})[ign], c,
                                                       pd.cut(d.r[ign], [0, 50, 100, 200, 400, 800, 5000]), "recall"), "recall")


if __name__ == "__main__":
    main()
