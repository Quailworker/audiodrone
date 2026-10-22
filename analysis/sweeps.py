"""Controlled sweeps: change one thing at a time and see where the model stops working.

    python -m analysis.sweeps --sweep range rpm confusers beds --out analysis/out/sweeps --workers 6

  range      each drone class flies a straight pass at a set closest distance, quiet and busier background
  rpm        electric drones hovering 30 m away with tip speed swept far beyond the training range
  confusers  each confuser type alone (no drone), at its natural levels and distances
  beds       each background type and the wind alone, at set levels (no drone)

Scenes are rendered with the same physics as sim/scene.py, then models B and A are run on them.
Writes <out>/<sweep>.pkl: one row per scene with its settings, probabilities and labels at 25 fps.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.dump import load_model
from cnn.data import frame_targets
from cnn.evaluate import predict
from sim import acoustics as ac
from sim import sources as src
from sim.scene import (FS, PAD_S, _ctrl_times, drone_signal, frame_labels, rotor_shaft_hz,
                       sample_confuser, sample_drone, traj_static)

DURATION = 40.0
MODELS = {"B": ("models/B_logf_best.pt", "logf"), "A": ("models/A_mel_best.pt", "mel")}


def straight_pass(cpa, alt, speed, duration=DURATION):
    """Straight, level pass whose closest horizontal distance `cpa` is reached mid-clip."""
    t = _ctrl_times(duration)
    x = speed * (t - duration / 2)
    return t, np.stack([x, np.full_like(t, cpa), np.full_like(t, alt)], 1)


def _first_draw_kind(rng, kind):
    """The sub-type sample_confuser will pick, from its first random draw (mirrors sim/scene.py)."""
    if kind == "ground_2stroke":
        return "ground_2stroke" if rng.uniform() < 0.7 else "moped"
    if kind == "vehicle":
        return ["car", "motorbike", "mower"][int(rng.choice(3, p=[.55, .25, .2]))]
    if kind == "aircraft":
        return "helicopter" if rng.uniform() < 0.5 else "light_aircraft"
    return kind


def render(spec):
    """spec: dict with seed and optional drone / confuser / bed settings -> (audio, labels, spec)."""
    rng = np.random.default_rng(spec["seed"])
    mic = np.array([0.0, 0.0, spec.get("mic_h", 0.02)])
    env = dict(temp_c=20.0, rh=60.0, ground_r=spec.get("ground_r", 0.75))
    n_out, n_src = int(DURATION * FS), int((DURATION + PAD_S) * FS)
    per_cat, tracks = {}, []

    def add(cat, sig, t, p, directivity=None):
        y, _ = ac.propagate_point_source(sig, -PAD_S, FS, t, p, mic, n_out, env["ground_r"],
                                         env["temp_c"], env["rh"], directivity)
        per_cat[cat] = per_cat.get(cat, 0) + y[None]

    if "cls" in spec:
        q = sample_drone(rng, spec["cls"], False)
        for k in ("blades", "rotors", "level", "speed", "alt"):
            if k in spec:
                q[k] = spec[k]
        if "v_tip" in spec:
            q["v_tip"] = spec["v_tip"]
            q["rpm_hover"] = 60 * spec["v_tip"] / (np.pi * q["prop_in"] * 0.0254)
        if spec.get("hover"):
            t, p = traj_static([spec["cpa"], 0.0, q["alt"]], DURATION)
        else:
            t, p = straight_pass(spec["cpa"], q["alt"], q["speed"])
        shaft = rotor_shaft_hz(rng, q, t, p, spec["cls"] == "ts_fw")
        sig, _ = drone_signal(rng, q, shaft, t, n_src)
        beta = q["directivity_beta"]
        cat = "drone_electric" if q["power"] == "electric" else "drone_engine"
        add(cat, ac.set_level_dba(sig, FS, q["level"]), t, p, lambda tau, el, b=beta: (1 + b * np.sin(el) ** 2) / (1 + b / 2))
        tracks.append((q, t, p, shaft))
        spec = {**spec, "rpm_hover": q["rpm_hover"], "prop_in": q["prop_in"], "level": q["level"], "alt": q["alt"],
                "blades": q["blades"], "rotors": q["rotors"], "speed": q["speed"]}

    if "confuser" in spec:
        want, kind = spec["confuser"], spec["confuser_kind"]
        for k in range(1000):                    # find a seed whose first draw picks the wanted sub-type
            cseed = spec["seed"] + 1 + 7919 * k
            if _first_draw_kind(np.random.default_rng(cseed), kind) == want:
                break
        cat, sig, t, p, level, cmeta = sample_confuser(np.random.default_rng(cseed), kind, DURATION, n_src)
        assert cmeta["kind"] == want, (cmeta["kind"], want)
        add(cat, ac.set_level_dba(sig, FS, level), t, p)
        spec = {**spec, "confuser_level": level, "confuser_dist": float(np.linalg.norm(p[len(p) // 2, :2]))}

    bed = np.zeros(n_out, np.float32)
    for kind in spec.get("beds", ["quiet_rural"]):
        for _ in range(6):
            bed += src.bed(rng, kind, n_out, FS)
    bed *= ac.P_REF * 10 ** (spec.get("bed_level", 40) / 20) / max(ac.a_weighted_rms(bed, FS), 1e-20)
    if spec.get("wind_level"):
        bed += ac.set_level_dba(src.mic_wind(rng, n_out, FS), FS, spec["wind_level"])
    bed += ac.set_level_dba(rng.standard_normal(n_out).astype(np.float32), FS, 25)
    per_cat["bed"] = bed[None]
    audio = sum(per_cat.values())[0]
    labels = frame_labels(per_cat, tracks, mic, n_out)
    return (np.clip(audio / ac.FULL_SCALE_PA, -1, 1)).astype(np.float32), labels, spec


def specs_for(sweep):
    out = []
    if sweep == "range":
        alts = {"fpv": 30, "mid": 50, "ts_mr": 60, "ts_fw": 120}
        for cls, alt in alts.items():
            for cpa in [5, 25, 50, 100, 150, 200, 300, 450, 650, 900]:
                for bed_level in (40, 55):
                    for s in range(6):
                        out.append(dict(seed=10_000 * len(out) + s, cls=cls, cpa=cpa, alt=alt, bed_level=bed_level,
                                        beds=["quiet_rural"] if bed_level == 40 else ["urban"]))
    elif sweep == "rpm":
        for cls in ("fpv", "mid"):
            for v_tip in [50, 65, 80, 95, 110, 125, 140, 155, 170, 190, 210]:
                for s in range(6):
                    out.append(dict(seed=20_000_000 + 1000 * len(out) + s, cls=cls, v_tip=v_tip, cpa=30, alt=10,
                                    hover=True, bed_level=40))
    elif sweep == "confusers":
        kinds = {"chainsaw": "ground_2stroke", "moped": "ground_2stroke", "car": "vehicle", "motorbike": "vehicle",
                 "mower": "vehicle", "helicopter": "aircraft", "light_aircraft": "aircraft", "fan": "fan", "hum": "hum",
                 "insect": "insect", "bird": "bird", "crickets": "crickets"}
        for want, kind in kinds.items():
            for s in range(15):
                out.append(dict(seed=30_000_000 + 1000 * len(out) + s, confuser=want if want != "chainsaw" else "ground_2stroke",
                                confuser_kind=kind, label=want, bed_level=40))
    elif sweep == "beds":
        for bed in ["quiet_rural", "wind_trees", "traffic_rumble", "urban", "rain"]:
            for level in (40, 55, 70):
                for s in range(4):
                    out.append(dict(seed=40_000_000 + 1000 * len(out) + s, beds=[bed], bed_level=level, label=bed))
        for wind in (50, 60, 70, 80):
            for s in range(4):
                out.append(dict(seed=40_000_000 + 1000 * len(out) + s, beds=["quiet_rural"], bed_level=35, wind_level=wind,
                                label=f"wind {wind} dBA"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", nargs="+", default=["range", "rpm", "confusers", "beds"])
    ap.add_argument("--out", default="analysis/out/sweeps")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {k: load_model(p, f, "cnn14", device) for k, (p, f) in MODELS.items()}
    for sweep in a.sweep:
        specs = specs_for(sweep)
        rows = []
        with ProcessPoolExecutor(a.workers) as pool:
            for i, (audio, labels, spec) in enumerate(pool.map(render, specs)):
                n = len(labels["frame_t"])
                y, m = frame_targets(labels, n)
                idx = np.arange(2, n, 4)
                row = {**{k: v for k, v in spec.items() if not isinstance(v, list)}, "beds": "+".join(spec.get("beds", [])),
                       "t": idx / 100, "y": y[idx], "m": m[idx],
                       "r": labels["drone0/r"][idx] if "drone0/r" in labels else None,
                       "lines1": {k[10:]: labels[k][idx, 1].astype(np.float32) for k in labels if k.startswith("snr_lines/")}}
                for name, model in models.items():
                    row[f"p_{name}"] = predict(model, audio, device)[: n][idx].astype(np.float16)
                rows.append(row)
                if (i + 1) % 50 == 0:
                    print(f"{sweep} {i + 1}/{len(specs)}", flush=True)
        pd.to_pickle(rows, Path(a.out) / f"{sweep}.pkl")
        print(f"{sweep}: {len(rows)} scenes", flush=True)


if __name__ == "__main__":
    main()
