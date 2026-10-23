"""Render a hand-built scenario on real terrain: mic clusters and drone flights placed in the UAE tool.

Same sound models as the training scenes (sim/scene.py, sim/sources.py), with three terrain changes:

1. Heights: each cluster stands on the DEM ground (+ mount height). Flights are given above ground level
   (alt_mode "agl": terrain-following, smoothed over ~3 s as an autopilot would) or at a fixed altitude
   ("fixed": the first waypoint's ground + agl, raised to clear the ground by >= 5 m).
2. Ground echo: mirrored in the local ground plane under each cluster (least-squares fit within 20 m),
   not in z = 0, so a cluster on a slope hears the echo from the right direction.
3. Screening: hills and buildings between the drone and a cluster: single knife-edge diffraction
   (terra/dem.py: ITU-R P.526, capped at 25 dB as ISO 9613-2), frequency dependent, updated 10 times a
   second along the flight and applied to direct sound and echo alike.

Ambient sound is rendered per cluster (independent between clusters, plane waves over the cluster's
mics as in the training scenes). Clusters render in parallel processes.

Scenario (JSON from the tool, local metres around origin):
    {"origin": [lat, lon], "half_m": 2000, "terrain": true, "mount": "ground",
     "clusters": [[x, y], ...], "sites": [{"name", "x", "y", "radius_m"}],
     "drones": [{"cls": "mid", "waypoints": [[x, y], ...], "agl": 60, "speed": 12, "alt_mode": "agl"}],
     "ambient": "desert", "temp_c": 35, "rh": 30, "seed": 1}
"""
import multiprocessing as mp

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from sim import acoustics as ac
from sim import sources as src
from sim.layouts import MOUNT_Z, cluster_offsets
from sim.scene import (
    CTRL_HZ,
    FS,
    PAD_S,
    _ctrl_times,
    drone_signal,
    frame_labels,
    rotor_shaft_hz,
    sample_drone,
)

from .dem import Terrain, knife_edge_db

FS_OUT = 8000
FRAME_HZ = 10
MAX_DURATION = 300.0
DRONE_CLASSES = {"fpv": "FPV 5-7 in (electric)", "mid": "Multirotor 7-12 in (electric)",
                 "ts_mr": "Multirotor 10 in+ (two-stroke)", "ts_fw": "Fixed wing (two-stroke)"}
AMBIENT = {                        # bed kinds, bed level dB(A), mic wind dB(A) or None
    "desert": (["quiet_rural"], 35.0, None),
    "desert_wind": (["wind_trees"], 42.0, 52.0),
    "town": (["urban", "traffic_rumble"], 52.0, None),
    "highway": (["traffic_rumble"], 60.0, None),
}


class FlatGround:
    """Terrain switched off: flat ground at 0 m with the same interface as Terrain."""

    def __init__(self, half_m):
        self.half = half_m

    def height(self, x, y):
        return np.zeros(np.shape(x)) if np.ndim(x) else 0.0

    def ground_plane(self, x, y, radius=20.0):
        return 0.0, np.array([0.0, 0.0, 1.0])

    def screening(self, src_pts, rcv, **kw):
        n = len(np.atleast_2d(src_pts))
        return np.full(n, -1e3), np.zeros(n, bool)


def load_terrain(scn):
    if not scn.get("terrain", True):
        return FlatGround(scn.get("half_m", 2000.0))
    lat, lon = scn["origin"]
    return Terrain(lat, lon, half_m=scn.get("half_m", 2000.0) + 500.0, res_m=10.0)


def flight_path(terrain, wp, agl, speed, alt_mode="agl"):
    """Waypoints (n, 2) -> control-rate (t (N,), p (N, 3)), duration. Starts at the first waypoint at t = 0."""
    wp = np.asarray(wp, float)
    legs = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    times = np.r_[0.0, np.cumsum(legs / max(speed, 0.5))]
    duration = float(min(times[-1] + 5.0, MAX_DURATION))
    t = _ctrl_times(duration)
    tc = np.clip(t, 0, times[-1])
    xy = np.column_stack([np.interp(tc, times, wp[:, i]) for i in range(2)])
    xy = gaussian_filter1d(xy, 1.5 * CTRL_HZ, axis=0, mode="nearest")
    g = terrain.height(xy[:, 0], xy[:, 1])
    if alt_mode == "fixed":
        z = np.full(len(t), terrain.height(*wp[0]) + agl)
        z = np.maximum(z, g + 5.0)
    else:
        z = gaussian_filter1d(g, 3.0 * CTRL_HZ, mode="nearest") + agl
    z = np.maximum(z, g + 2.0)
    return t, np.column_stack([xy, z]), duration


def build(scn):
    """Everything the cluster renders share: terrain, cluster geometry, drone flights and source signals."""
    terrain = load_terrain(scn)
    mount = MOUNT_Z[scn.get("mount", "ground")]
    off = cluster_offsets("flat")
    centres, planes = [], []
    for x, y in scn["clusters"]:
        z0, n = terrain.ground_plane(x, y)
        centres.append([x, y, terrain.height(x, y) + mount])
        planes.append((np.array([x, y, z0]), n))
    centres = np.array(centres)
    rng_env = np.random.default_rng([scn.get("seed", 1), 99])
    env = {"temp_c": float(scn.get("temp_c", 35.0)), "rh": float(scn.get("rh", 30.0)),
               "ground_r": float(rng_env.uniform(0.7, 0.95))}          # desert sand and rock: hard-ish ground
    drones, duration = [], 5.0
    for d in scn["drones"]:
        t, p, dur = flight_path(terrain, d["waypoints"], d.get("agl", 60.0), d.get("speed", 12.0), d.get("alt_mode", "agl"))
        drones.append((d, t, p))
        duration = max(duration, dur)
    n_out, n_src = int(duration * FS), int((duration + PAD_S) * FS)
    sources, tracks = [], []
    for i, (d, t, p) in enumerate(drones):
        if t[-1] < duration + 1:                                       # hover at the last point to the end
            extra = np.arange(1, int((duration + 1 - t[-1]) * CTRL_HZ) + 2) / CTRL_HZ
            t = np.r_[t, t[-1] + extra]
            p = np.vstack([p, np.repeat(p[-1:], len(extra), 0)])
        rng = np.random.default_rng([scn.get("seed", 1), i])
        q = sample_drone(rng, d["cls"], False)
        q["speed"] = float(d.get("speed", 12.0))
        if "level" in d:
            q["level"] = float(d["level"])
        shaft = rotor_shaft_hz(rng, q, t, p, d["cls"] == "ts_fw")
        sig, _ = drone_signal(rng, q, shaft, t, n_src)
        sig = ac.set_level_dba(sig, FS, q["level"])
        cat = "drone_electric" if q["power"] == "electric" else "drone_engine"
        sources.append((cat, sig, t, p, q["directivity_beta"]))
        tracks.append((q, t, p, shaft))
    return {"terrain": terrain, "centres": centres, "planes": planes, "offsets": off, "mount": mount, "env": env,
                "duration": duration, "n_out": n_out, "sources": sources, "tracks": tracks}


def _directivity(beta):
    return lambda tau, el: (1 + beta * np.sin(el) ** 2) / (1 + beta / 2)


def screening_track(terrain, t_ctrl, p_ctrl, rcv, tf):
    """Knife-edge factor g and blocked flag at receive times tf, from the emission-time position."""
    tau, _ = ac.emission_time(tf, t_ctrl, p_ctrl, rcv)
    return terrain.screening(ac.interp_path(t_ctrl, p_ctrl, tau), rcv)


def render_cluster(args):
    """One cluster: 4 mics at 8 kHz, the reference mic at 32 kHz (for the CNN), labels and screening."""
    k, shared, scn = args
    terrain, env, n_out = shared["terrain"], shared["env"], shared["n_out"]
    centre = shared["centres"][k]
    ground_pt, normal = shared["planes"][k]
    mics = centre[None] + shared["offsets"]
    tf = np.arange(0, shared["duration"], 1 / FRAME_HZ) + 0.5 / FRAME_HZ
    per_cat, screen = {}, []
    for cat, sig, t, p, beta in shared["sources"]:
        g, blocked = screening_track(terrain, t, p, centre, tf)
        screen.append({"g": g.astype(np.float32), "blocked": blocked})
        def loss(f, tt, g=g):
            return knife_edge_db(np.interp(tt, tf, g)[None, :] * np.sqrt(np.maximum(f, 1.0)[:, None] / ac.C_SOUND))
        rows = []
        for m in mics:
            image = m - 2 * ((m - ground_pt) @ normal) * normal
            y, _ = ac.propagate_point_source(sig, -PAD_S, FS, t, p, m, n_out, env["ground_r"], env["temp_c"],
                                             env["rh"], _directivity(beta), image=image, extra_loss_db=loss)
            rows.append(y)
        per_cat[cat] = per_cat.get(cat, 0) + np.array(rows)
    # ambient: independent per cluster, plane waves across the cluster's mics, then wind and self-noise
    kinds, bed_level, wind_level = AMBIENT[scn.get("ambient", "desert")]
    rng = np.random.default_rng([scn.get("seed", 1), 7, k])
    bed = np.zeros((4, n_out), np.float32)
    off = shared["offsets"]
    for kind in kinds:
        for _ in range(6):
            x = src.bed(rng, kind, n_out, FS)
            az, el = rng.uniform(0, 2 * np.pi), rng.uniform(0, 0.4)
            u = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
            for j in range(4):
                bed[j] += ac.fractional_shift(x, -(u @ off[j]) / ac.C_SOUND, FS)
    bed *= ac.P_REF * 10 ** (bed_level / 20) / max(ac.a_weighted_rms(bed[0], FS), 1e-20)
    for j in range(4):
        if wind_level:
            bed[j] += ac.set_level_dba(src.mic_wind(rng, n_out, FS), FS, wind_level)
        bed[j] += ac.set_level_dba(rng.standard_normal(n_out).astype(np.float32), FS, rng.uniform(20, 28))
    per_cat["bed"] = bed
    audio = sum(per_cat.values()) / ac.FULL_SCALE_PA
    labels = frame_labels({c: x[[0]] for c, x in per_cat.items()}, shared["tracks"], mics[0], n_out)
    x8 = signal.resample_poly(audio, 1, FS // FS_OUT, axis=1).astype(np.float32)
    return k, np.clip(x8, -1, 1), np.clip(audio[0], -1, 1).astype(np.float32), labels, screen


def render(scn, workers=6, progress=None):
    """-> (shared, list of per-cluster results in cluster order)."""
    shared = build(scn)
    jobs = [(k, shared, scn) for k in range(len(shared["centres"]))]
    out = [None] * len(jobs)
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(workers, len(jobs))) as pool:
        for i, r in enumerate(pool.imap_unordered(render_cluster, jobs)):
            out[r[0]] = r
            if progress:
                progress(f"rendered cluster {r[0] + 1} ({i + 1}/{len(jobs)})")
    return shared, out
