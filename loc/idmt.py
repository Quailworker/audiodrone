"""Direction finding on the real Fraunhofer IDMT array recordings vs GPS (task step 5; open question 25).

    python -m loc.idmt --root data/real/idmt --out results/loc_v1/idmt

Data: Zenodo 17525811 (CC BY 4.0), test only. 163 x 30 s segments, 8 channels at 48 kHz, plus
DOATruePos.csv (time, X, Y, Z, ele, azi) from GPS loggers (+-2.5 m, 10 Hz) on the drone and the array.
The GPS log has no altitude (Z = 0), so only **azimuth** can be checked against truth.

Array geometry: Table 1 of Yang et al., "Improving acoustic drone detection generalization through
pretraining and data augmentation" (arXiv 2605.31329): 8 mics on a 0.15 m cube, positions relative to
mic 8; x north, z up, y by the right-hand rule. Channel i of the WAV is taken to be mic i + 1 (the
dataset does not state the channel order; the azimuth errors below would be large if it were wrong).
Largest spacing ~0.21 m, so the direction-finding band stops at 800 Hz (c / 2d).

Heading calibration: the estimated azimuths match the GPS azimuths after a constant rotation of about
210 deg (no mirroring) in every flight with a wide azimuth sweep, so geometry and channel order are right
and the array's x axis was not aligned with the GPS frame's x axis (or the GPS azimuth is measured from
the drone to the array plus a ~30 deg heading offset). design.md 7.7 calls for calibrating each array with
a GPS-logged flight; here the rotation for each flight is estimated from the *other* flights only
(leave-one-flight-out, frames with coh >= 0.4 and rho >= 30 m), and both raw and calibrated errors are
reported (<m>_err, <m>_err_cal).

Truth at each frame is evaluated at the emission time t - r/c. GPS error alone gives an azimuth error of
about atan(2.5 / rho): ~7 deg at 20 m, ~1.4 deg at 100 m.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

from .doa import DOA, BeamformerDOA

MICS = np.array([[0.05625, 0.03750, 0.15000], [0.00000, 0.09000, 0.15000], [0.01875, 0.15000, 0.11450],
                 [0.05625, 0.15000, 0.00900], [0.15000, 0.15000, 0.07500], [0.13125, 0.07500, 0.00900],
                 [0.15000, 0.01875, 0.15000], [0.00000, 0.00000, 0.00000]])
FS = 8000
BAND = (100.0, 800.0)


def az_err(a, b):
    return np.abs((a - b + 180) % 360 - 180)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/real/idmt")
    ap.add_argument("--out", default="results/loc_v1/idmt")
    a = ap.parse_args()
    off = MICS - MICS.mean(0)
    est = {"srp": DOA(off, FS, band=BAND), "srp_beta1": DOA(off, FS, band=BAND, phat_beta=1.0),
           "bf_free": BeamformerDOA(off, FS, height=1.0, refl=0.0, band=BAND)}
    rows = []
    segs = sorted(p for p in (Path(a.root) / "MicArrayData30s").iterdir() if p.is_dir())
    for i, seg in enumerate(segs):
        x, sr = sf.read(next(seg.glob("*.wav")), dtype="float32", always_2d=True)
        g = np.gcd(sr, FS)
        x = signal.resample_poly(x.T, FS // g, sr // g, axis=1).astype(np.float32)
        pos = pd.read_csv(seg / "DOATruePos.csv")
        tf = np.arange(0.25, x.shape[1] / FS - 0.25, 0.1)
        rho = np.interp(tf, pos.time, np.hypot(pos.X, pos.Y))
        te = tf - rho / 343.0                                                    # emission time (level flight)
        azi = np.degrees(np.unwrap(np.radians(pos.azi.to_numpy())))
        true_az = np.interp(te, pos.time, azi) % 360
        flight = seg.name.rsplit("_", 2)[0]
        row = pd.DataFrame({"segment": seg.name, "t": tf, "rho": rho, "true_az": true_az,
                            "drone": next(k for k in ("Phantom4", "Mavic", "Align") if k in flight),
                            "kind": next(k for k in ("static", "linear", "random") if k in flight)})
        for name, e in est.items():
            r = e.run(x, tf) if name.startswith("bf") else e.run(x, tf, methods=("srp", "ls"), chunk=16)
            for m, res in ([(name, r)] if name.startswith("bf") else [(name, r["srp"])] + ([("ls", r["ls"])] if name == "srp" else [])):
                row[f"{m}_az"], row[f"{m}_el"], row[f"{m}_coh"] = res["az"], res["el"], res["coh"]
                row[f"{m}_err"] = az_err(res["az"], true_az)
        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(segs)}", flush=True)
    df = pd.concat(rows, ignore_index=True)
    df["flight"] = df.segment.str.rsplit("_", n=2).str[0]
    cal = {}
    for m in [c[:-3] for c in df.columns if c.endswith("_az") and c != "true_az"]:
        for fl in df.flight.unique():
            ref = df[(df.flight != fl) & (df[f"{m}_coh"] >= 0.4) & (df.rho >= 30)]
            rot = float(np.degrees(np.angle(np.exp(1j * np.radians(ref[f"{m}_az"] - ref.true_az)).mean())))
            cal.setdefault(m, {})[fl] = rot
            sel = df.flight == fl
            df.loc[sel, f"{m}_err_cal"] = az_err(df.loc[sel, f"{m}_az"] - rot, df.loc[sel, "true_az"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "idmt_frames.csv.gz", index=False, float_format="%.3f")
    summary = summarise(df)
    summary["heading_rotation_deg_lofo"] = cal
    (out / "idmt_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["by_range"], indent=1))


def summarise(df, methods=("srp", "srp_beta1", "ls", "bf_free")):
    bins = [0, 25, 50, 100, 150, 200, 450]
    df = df.assign(rb=pd.cut(df.rho, bins).astype(str))
    out = {"frames": int(len(df)), "by_range": {}, "by_drone": {}, "coh_gate": {}}
    for m in methods:
        for col in (f"{m}_err", f"{m}_err_cal"):
            g = df.groupby("rb", observed=True)[col]
            out["by_range"][col] = {k: {"n": int(len(v)), "median": float(v.median()), "p90": float(v.quantile(0.9)),
                                        "gross_gt10": float((v > 10).mean())} for k, v in g}
        out["by_drone"][m] = {k: float(v.median()) for k, v in df[df.rho <= 100].groupby("drone")[f"{m}_err_cal"]}
        for q in (0.0, 0.5, 0.75, 0.9):
            thr = float(df[f"{m}_coh"].quantile(q))
            sel = df[(df[f"{m}_coh"] >= thr) & (df.rho <= 100)]
            out["coh_gate"].setdefault(m, {})[f"top{int(100 * (1 - q))}pct"] = {
                "coh_min": thr, "median_err_cal_le100m": float(sel[f"{m}_err_cal"].median()),
                "gross_gt10": float((sel[f"{m}_err_cal"] > 10).mean())}
    return out


if __name__ == "__main__":
    main()
