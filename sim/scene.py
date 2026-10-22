"""Scenario sampling and rendering (design.md 14.6).

A scene = one microphone (or several), a set of sources with trajectories, and diffuse beds.
render_scene() returns the mic audio in pascals plus per-frame labels.
"""
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d, uniform_filter

from . import acoustics as ac
from . import sources as src

FS = 32000
CTRL_HZ = 50
PAD_S = 10.0                       # source history needed for the longest propagation delay
G = 9.81
HOP = 320                          # label frame hop: 100 frames/s, same as CNN14's input
CATEGORIES = ["drone_electric", "drone_engine", "ground_engine", "vehicle", "aircraft",
              "fan", "hum", "insect", "bird", "bed"]
TRAIN_BEDS = ["wind_trees", "traffic_rumble", "urban", "quiet_rural"]
HELDOUT_BEDS = ["rain"]


# --- trajectories ------------------------------------------------------------------
# Each returns (t_ctrl, positions (N, 3), duration_s). t_ctrl spans [-PAD_S, duration].

def _ctrl_times(duration):
    return np.arange(-PAD_S, duration + 1.0, 1 / CTRL_HZ)


def _wobble(rng, t, cutoff, amp):
    n = len(t)
    return src.smooth_noise(rng, n, CTRL_HZ, cutoff, amp) if amp > 0 else np.zeros(n, np.float32)


def traj_flyover(rng, speed, alt, r_far, max_cpa=300.0):
    # most passes come close (the whole pass should be heard); some stay far
    d = rng.uniform(0, 60) if rng.uniform() < 0.6 else rng.uniform(60, max(60.0, min(max_cpa, 0.8 * r_far)))
    half = np.sqrt(max(r_far ** 2 - d ** 2 - alt ** 2, 100.0 ** 2))
    duration = float(np.clip(2 * half / speed, 20, 120))
    half = speed * duration / 2
    t = _ctrl_times(duration)
    s = -half + speed * t + np.cumsum(_wobble(rng, t, 0.2, 0.1 * speed)) / CTRL_HZ
    lat = d + _wobble(rng, t, 0.05, rng.uniform(0, 10))
    z = np.maximum(alt + _wobble(rng, t, 0.1, rng.uniform(0, 0.1 * alt + 1)), 1.0)
    psi = rng.uniform(0, 2 * np.pi)
    x = s * np.cos(psi) - lat * np.sin(psi)
    y = s * np.sin(psi) + lat * np.cos(psi)
    return t, np.stack([x, y, z], 1), duration, {"family": "flyover", "cpa_m": d, "speed": speed, "alt": alt}


def _path_through(rng, points, speeds, dwell=None):
    """Piecewise-linear path through points at given leg speeds, optional dwell at index."""
    times, pos = [0.0], [points[0]]
    for i in range(1, len(points)):
        leg = np.linalg.norm(points[i] - points[i - 1])
        times.append(times[-1] + leg / speeds[i - 1])
        pos.append(points[i])
        if dwell and dwell[0] == i:
            times.append(times[-1] + dwell[1])
            pos.append(points[i])
    return np.array(times), np.array(pos)


def _sample_path(rng, times, pos, max_duration=120.0, smooth_s=1.5):
    duration = float(min(times[-1], max_duration))
    t = _ctrl_times(duration)
    tc = np.clip(t, 0, times[-1])
    p = np.stack([np.interp(tc, times, pos[:, i]) for i in range(3)], 1)
    p = gaussian_filter1d(p, smooth_s * CTRL_HZ, axis=0, mode="nearest")
    p[:, 2] = np.maximum(p[:, 2], 1.0)
    return t, p, duration


def traj_approach_hover(rng, speed, alt, r_far):
    dwell = rng.uniform(5, 40)
    leg = min(r_far, (120 - dwell) / 2 * speed)
    a0, a1 = rng.uniform(0, 2 * np.pi, 2)
    hover = np.array([*rng.uniform(-100, 100, 2), alt * rng.uniform(0.5, 1.2)])
    pts = np.array([[leg * np.cos(a0), leg * np.sin(a0), alt], hover,
                    [leg * np.cos(a1), leg * np.sin(a1), alt * rng.uniform(0.5, 1.5)]])
    times, pos = _path_through(rng, pts, [speed, speed], dwell=(1, dwell))
    t, p, duration = _sample_path(rng, times, pos)
    return t, p, duration, {"family": "approach_hover", "hover_s": dwell, "speed": speed, "alt": alt}


def traj_orbit(rng, speed, alt, r_far):
    radius = rng.uniform(20, 150)
    centre = rng.uniform(-150, 150, 2)
    duration = rng.uniform(30, 90)
    t = _ctrl_times(duration)
    ang = rng.uniform(0, 2 * np.pi) + rng.choice([-1, 1]) * speed / radius * t
    z = np.maximum(alt + _wobble(rng, t, 0.05, 0.1 * alt + 1), 1.0)
    p = np.stack([centre[0] + radius * np.cos(ang), centre[1] + radius * np.sin(ang), z], 1)
    return t, p, duration, {"family": "orbit", "radius": radius, "speed": speed, "alt": alt}


def traj_waypoints(rng, speed, alt, r_far):
    n = int(rng.integers(4, 8))
    box = min(r_far, 250)
    pts = np.column_stack([rng.uniform(-box, box, (n, 2)), np.clip(alt * rng.uniform(0.3, 1.7, n), 2, None)])
    near = int(rng.integers(1, n - 1))            # route always passes near the mic once
    pts[near, :2] = rng.uniform(-50, 50, 2)
    times, pos = _path_through(rng, pts, speed * rng.uniform(0.6, 1.3, n - 1))
    t, p, duration = _sample_path(rng, times, pos)
    if duration < 20:
        return traj_flyover(rng, speed, alt, r_far)
    return t, p, duration, {"family": "waypoints", "speed": speed, "alt": alt}


def traj_static(pos, duration):
    t = _ctrl_times(duration)
    return t, np.repeat(np.asarray(pos, float)[None], len(t), 0)


def traj_line(rng, duration, dist, speed, height):
    """Ground vehicle on a straight road at perpendicular distance dist, passing at a random time."""
    t = _ctrl_times(duration)
    t_pass = rng.uniform(0.1, 0.9) * duration
    psi = rng.uniform(0, 2 * np.pi)
    s = speed * (t - t_pass)
    x = s * np.cos(psi) - dist * np.sin(psi)
    y = s * np.sin(psi) + dist * np.cos(psi)
    return t, np.stack([x, y, np.full_like(t, height)], 1)


def kinematics(t, p):
    v = np.gradient(p, t, axis=0)
    a = gaussian_filter1d(np.gradient(v, t, axis=0), 0.3 * CTRL_HZ, axis=0, mode="nearest")
    return v, a


# --- drone classes (design.md 4.4) --------------------------------------------------

def _choice(rng, options, probs):
    return options[int(rng.choice(len(options), p=probs))]


def sample_drone(rng, cls, heldout):
    """Physical + sound parameters for one drone of class cls."""
    if cls == "fpv":
        d_in = rng.uniform(5, 7)
        q = dict(power="electric", blades=_choice(rng, [2, 3, 4], [.15, .7, .15]),
                 rotors=_choice(rng, [5] if heldout else [4, 6], [1] if heldout else [.9, .1]),
                 v_tip=rng.uniform(135, 160) if heldout else rng.uniform(90, 130),
                 pole_pairs=_choice(rng, [6, 7], [.3, .7]), speed=rng.uniform(5, 40),
                 alt=rng.uniform(2, 120), level=rng.uniform(80, 95), r_far=rng.uniform(250, 450))
    elif cls == "mid":
        d_in = rng.uniform(7, 12)
        q = dict(power="electric", blades=_choice(rng, [2, 3], [.65, .35]),
                 rotors=_choice(rng, [7] if heldout else [4, 6, 8], [1] if heldout else [.6, .25, .15]),
                 v_tip=rng.uniform(125, 150) if heldout else rng.uniform(80, 120),
                 pole_pairs=_choice(rng, [7, 10, 12], [.5, .3, .2]), speed=rng.uniform(1, 20),
                 alt=rng.uniform(5, 200), level=rng.uniform(75, 92), r_far=rng.uniform(250, 450))
    elif cls == "ts_fw":
        d_in = rng.uniform(10, 26)
        q = dict(power="engine", blades=_choice(rng, [2, 3], [.8, .2]), rotors=1,
                 v_tip=rng.uniform(225, 250) if heldout else rng.uniform(140, 220),
                 speed=rng.uniform(20, 45), alt=rng.uniform(30, 400), level=rng.uniform(92, 108),
                 r_far=rng.uniform(900, 2000))
    elif cls == "ts_mr":
        d_in = rng.uniform(14, 32)
        q = dict(power="engine", blades=2, rotors=_choice(rng, [1, 2, 4], [.4, .2, .4]),
                 v_tip=rng.uniform(185, 210) if heldout else rng.uniform(120, 180),
                 speed=rng.uniform(1, 20), alt=rng.uniform(5, 200), level=rng.uniform(90, 105),
                 r_far=rng.uniform(600, 1500))
    else:
        raise ValueError(cls)
    q.update(cls=cls, prop_in=d_in)
    rpm = 60 * q["v_tip"] / (np.pi * d_in * 0.0254)
    if q["power"] == "engine":
        rpm = float(np.clip(rpm, 3000, 16000 if heldout else 14000))
    q["rpm_hover"] = rpm
    q["directivity_beta"] = rng.uniform(-0.4, 0.8)
    return q


def rotor_shaft_hz(rng, q, t, p, fixed_wing):
    """Shaft rate per rotor (n_rotors, len(t)) at control rate, from flight dynamics."""
    v, a = kinematics(t, p)
    speed = np.linalg.norm(v[:, :2], axis=1)
    if fixed_wing:
        climb = v[:, 2]
        throttle = 1 + 0.25 * climb / 5 + 0.2 * (speed - speed.mean()) / max(speed.mean(), 1)
        base = q["rpm_hover"] / 60 * np.clip(throttle, 0.6, 1.3)
        return (base * (1 + _wobble(rng, t, 0.5, 0.01)))[None]
    # multirotor: thrust balances gravity, acceleration and drag (tilt ~35 deg at top speed)
    c_drag = G * np.tan(np.radians(35)) / max(q["speed"], 1) ** 2
    thrust = a + np.array([0, 0, G]) + c_drag * v * np.linalg.norm(v, axis=1, keepdims=True)
    tf = np.linalg.norm(thrust, axis=1) / G
    tilt = np.arccos(np.clip(thrust[:, 2] / np.linalg.norm(thrust, axis=1), -1, 1))
    heading = np.arctan2(v[:, 1], v[:, 0])
    base = q["rpm_hover"] / 60 * np.sqrt(np.clip(tf, 0.3, 3.5))
    n = q["rotors"]
    split = rng.uniform(0.05, 0.15)
    out = []
    for i in range(n):
        ang = 2 * np.pi * i / n
        rel = -split * np.sin(tilt) * np.cos(ang - heading)           # rear rotors faster
        static = rng.normal(0, 0.015)
        ctrl = _wobble(rng, t, rng.uniform(1, 5), rng.uniform(0.005, 0.02))
        out.append(base * (1 + rel + static + ctrl))
    return np.array(out)


def drone_signal(rng, q, shaft_ctrl, t_ctrl, n_src):
    """Source signal at source rate for a drone (arbitrary units)."""
    t_src = -PAD_S + np.arange(n_src) / FS
    shaft = np.array([np.interp(t_src, t_ctrl, s) for s in shaft_ctrl]).astype(np.float32)
    if q["power"] == "electric":
        p = dict(slope=rng.uniform(0.8, 1.8), imbalance=rng.uniform(0.02, 0.3), whine=rng.uniform(0.03, 0.4),
                 phase_wander=rng.uniform(0.05, 0.6), flutter_db=rng.uniform(0.5, 3),
                 broadband=rng.uniform(0.1, 0.8), bb_mod=rng.uniform(0.2, 0.9))
        if rng.uniform() < 0.3:
            p.update(pwm_hz=rng.uniform(7000, 12000), pwm=rng.uniform(0.002, 0.02))
        return src.electric_rotors(rng, shaft, FS, q["blades"], q["pole_pairs"], p), p
    p = engine_params(rng)
    prop_rel = rng.uniform(0.2, 1.0)
    out = np.zeros(n_src, np.float32)
    for s in shaft:
        out += src.engine(rng, s, FS, 1.0, p)
        prop = src.propeller(rng, s, FS, q["blades"], rng.uniform(0.8, 1.6))
        out += prop_rel * prop / (prop.std() + 1e-12)
    return out, {**p, "prop_rel": prop_rel}


def engine_params(rng, muffled_prob=0.5):
    return dict(res_hz=rng.uniform(80, 500), q=rng.uniform(2, 8), click=rng.uniform(0, 1),
                muffler_hz=rng.uniform(800, 3000) if rng.uniform() < muffled_prob else None,
                jitter=rng.uniform(0.005, 0.03), cycle_var=rng.uniform(0.05, 0.25),
                misfire=rng.uniform(0, 0.02), mech=rng.uniform(0.02, 0.2))


# --- confusers ------------------------------------------------------------------------

def _ground_pos(rng, lo, hi, height=(0.3, 1.5)):
    d = np.exp(rng.uniform(np.log(lo), np.log(hi)))
    a = rng.uniform(0, 2 * np.pi)
    return [d * np.cos(a), d * np.sin(a), rng.uniform(*height)]


def _rev_profile(rng, n, fs, idle, high):
    """Stop-start revving (chainsaw, blower): smooth steps between idle and high RPM."""
    seg = int(rng.uniform(1.5, 8) * fs)
    levels = rng.choice([0, 1], size=n // seg + 2, p=[0.4, 0.6]) * rng.uniform(0.7, 1.0, n // seg + 2)
    curve = np.repeat(levels, seg)[:n]
    curve = gaussian_filter1d(curve.astype(np.float32), 0.15 * fs)
    return (idle + (high - idle) * curve) / 60


def sample_confuser(rng, kind, duration, n_src):
    """Returns (category, signal, t_ctrl, p_ctrl, level_db_at_1m, meta)."""
    meta = {"kind": kind}
    if kind == "ground_2stroke":       # chainsaw / leaf blower / trimmer, or a moped on a road
        if rng.uniform() < 0.7:
            shaft = _rev_profile(rng, n_src, FS, rng.uniform(2600, 3500), rng.uniform(8000, 12000))
            t, p = traj_static(_ground_pos(rng, 15, 500), duration)
            level = rng.uniform(100, 112)
        else:
            shaft = (rng.uniform(4000, 8000) / 60) * (1 + src.smooth_noise(rng, n_src, FS, 0.3, 0.1))
            t, p = traj_line(rng, duration, rng.uniform(10, 300), rng.uniform(8, 15), 0.6)
            level = rng.uniform(95, 105)
            meta["kind"] = "moped"
        sig = src.engine(rng, shaft.astype(np.float32), FS, 1.0, engine_params(rng, 0.3))
        return "ground_engine", sig, t, p, level, meta
    if kind == "vehicle":
        sub = _choice(rng, ["car", "motorbike", "mower"], [.55, .25, .2])
        meta["kind"] = sub
        if sub == "mower":
            shaft = np.full(n_src, rng.uniform(2800, 3400) / 60, np.float32)
            sig = src.engine(rng, shaft, FS, 0.5, engine_params(rng, 0.6))
            t, p = traj_static(_ground_pos(rng, 10, 300), duration)
            return "vehicle", sig, t, p, rng.uniform(88, 98), meta
        speed = rng.uniform(8, 30)
        fires = 2.0 if sub == "car" else _choice(rng, [0.5, 1.0], [.5, .5])
        shaft = (rng.uniform(1200, 3500) / 60) * (1 + src.smooth_noise(rng, n_src, FS, 0.2, 0.08))
        sig = src.engine(rng, shaft.astype(np.float32), FS, fires, {**engine_params(rng, 0.9), "cyl_imbalance": rng.uniform(0, 0.3)})
        tyre_rel = (speed / 20) ** 2 * (2.0 if sub == "car" else 0.5)
        sig = sig + tyre_rel * src.tyre_noise(rng, n_src, FS)
        t, p = traj_line(rng, duration, rng.uniform(8, 300), speed, 0.5)
        return "vehicle", sig, t, p, rng.uniform(85, 100), meta
    if kind == "aircraft":
        if rng.uniform() < 0.5:        # helicopter: main-rotor slap + tail rotor
            rotor_rpm = rng.uniform(250, 450)
            blades = int(rng.integers(2, 6))
            shaft = np.full(n_src, rotor_rpm / 60, np.float32)
            slap = src.engine(rng, shaft, FS, blades, dict(engine_params(rng, 1.0), res_hz=rng.uniform(40, 120), q=2))
            tail = src.propeller(rng, shaft * rng.uniform(4.5, 6), FS, int(rng.integers(2, 5)), 1.2)
            sig = slap + rng.uniform(0.2, 0.8) * tail / (tail.std() + 1e-12)
            level, alt, speed, meta["kind"] = rng.uniform(120, 133), rng.uniform(100, 600), rng.uniform(30, 70), "helicopter"
        else:                          # light aircraft: 4-cylinder four-stroke + 2-blade prop
            shaft = (rng.uniform(2300, 2700) / 60) * (1 + src.smooth_noise(rng, n_src, FS, 0.1, 0.01))
            shaft = shaft.astype(np.float32)
            sig = src.engine(rng, shaft, FS, 2.0, engine_params(rng, 0.8))
            prop = src.propeller(rng, shaft, FS, 2, 1.0)
            sig = sig + prop / (prop.std() + 1e-12)
            level, alt, speed, meta["kind"] = rng.uniform(118, 130), rng.uniform(300, 1500), rng.uniform(50, 70), "light_aircraft"
        cpa = rng.uniform(0, 1500)
        t, p = traj_line(rng, duration, cpa, speed, alt)
        meta.update(cpa_m=cpa, alt=alt, speed=speed)
        return "aircraft", sig, t, p, level, meta
    if kind == "fan":
        t, p = traj_static(_ground_pos(rng, 5, 80, (0.5, 4)), duration)
        return "fan", src.fan(rng, n_src, FS), t, p, rng.uniform(65, 82), meta
    if kind == "hum":
        t, p = traj_static(_ground_pos(rng, 3, 100, (0.5, 5)), duration)
        return "hum", src.mains_hum(rng, n_src, FS), t, p, rng.uniform(55, 85), meta
    if kind == "insect":
        n = len(_ctrl_times(duration))
        walk = np.cumsum(rng.normal(0, 0.05, (n, 3)), 0)
        centre = np.array(_ground_pos(rng, 0.3, 2.5, (0.1, 1.5)))
        p = centre + walk - walk.mean(0)
        p[:, 2] = np.clip(p[:, 2], 0.05, 3)
        t = _ctrl_times(duration)
        return "insect", src.insect(rng, n_src, FS), t, p, rng.uniform(20, 45), meta
    if kind == "bird":
        t, p = traj_static(_ground_pos(rng, 5, 80, (3, 15)), duration)
        return "bird", src.birds(rng, n_src, FS), t, p, rng.uniform(70, 90), meta
    if kind == "crickets":
        t, p = traj_static(_ground_pos(rng, 2, 30, (0, 0.3)), duration)
        return "insect", src.crickets(rng, n_src, FS), t, p, rng.uniform(60, 80), meta
    raise ValueError(kind)


CONFUSER_PROBS = {"ground_2stroke": .2, "vehicle": .35, "aircraft": .12, "fan": .15, "hum": .12,
                  "insect": .12, "bird": .35, "crickets": .2}
DRONE_CLASSES = ["fpv", "mid", "ts_fw", "ts_mr"]
DRONE_CLASS_PROBS = [.3, .3, .25, .15]


# --- scene -------------------------------------------------------------------------------

def sample_and_render(seed, heldout=False, mics=((0.0, 0.0, None),)):
    """Sample a random scene from seed and render it. Returns (audio (n_mics, n), labels, meta)."""
    rng = np.random.default_rng(seed)
    has_drone = rng.uniform() < 0.75
    n_drones = (2 if rng.uniform() < 0.1 else 1) if has_drone else 0

    mic_h = rng.uniform(0.01, 0.05) if rng.uniform() < 0.7 else rng.uniform(1, 3)
    mic_pos = np.array([[x, y, mic_h if z is None else z] for x, y, z in mics])
    env = dict(temp_c=rng.uniform(0, 35), rh=rng.uniform(20, 95), ground_r=rng.uniform(0.5, 1.0))
    meta = {"seed": int(seed), "heldout": heldout, "mic_pos": mic_pos.tolist(), **env, "sources": []}

    # drones first: they set the scene duration
    drones = []
    for _ in range(n_drones):
        cls = _choice(rng, DRONE_CLASSES, DRONE_CLASS_PROBS)
        q = sample_drone(rng, cls, heldout)
        fixed_wing = cls == "ts_fw"
        families = [traj_flyover, traj_orbit, traj_waypoints] if fixed_wing else \
                   [traj_flyover, traj_approach_hover, traj_orbit, traj_waypoints]
        probs = [.6, .2, .2] if fixed_wing else [.5, .2, .1, .2]
        fam = families[int(rng.choice(len(families), p=probs))]
        t, p, dur, info = fam(rng, q["speed"], q["alt"], q["r_far"])
        drones.append((q, fixed_wing, t, p, dur, info))
    duration = max([d[4] for d in drones], default=rng.uniform(30, 120))
    n_out = int(duration * FS)
    n_src = int((duration + PAD_S) * FS)

    per_cat = {}                       # category -> (n_mics, n_out) pascals
    drone_tracks = []

    def add(cat, sig_pa, t, p, directivity=None):
        rows, rs = [], []
        for m in mic_pos:
            y, r = ac.propagate_point_source(sig_pa, -PAD_S, FS, t, p, m, n_out, env["ground_r"],
                                             env["temp_c"], env["rh"], directivity)
            rows.append(y)
            rs.append(r)
        per_cat[cat] = per_cat.get(cat, 0) + np.array(rows)
        return rs

    for q, fixed_wing, t, p, _dur, info in drones:
        t = np.concatenate([t, t[-1] + np.arange(1, int((duration - t[-1] + 2) * CTRL_HZ) + 1) / CTRL_HZ]) \
            if t[-1] < duration + 1 else t
        # a shorter flight continues at its final velocity rather than stopping dead
        v_end = (p[-1] - p[-1 - CTRL_HZ]) / 1.0
        steps = np.arange(1, len(t) - len(p) + 1)[:, None] / CTRL_HZ
        p = np.concatenate([p, p[-1] + v_end * steps])
        p[:, 2] = np.maximum(p[:, 2], 1.0)
        shaft = rotor_shaft_hz(rng, q, t, p, fixed_wing)
        sig, sound = drone_signal(rng, q, shaft, t, n_src)
        sig = ac.set_level_dba(sig, FS, q["level"])
        beta = q["directivity_beta"]
        def directivity(tau, el, b=beta):
            return (1 + b * np.sin(el) ** 2) / (1 + b / 2)
        cat = "drone_electric" if q["power"] == "electric" else "drone_engine"
        add(cat, sig, t, p, directivity)
        drone_tracks.append((q, t, p, shaft))
        meta["sources"].append({"category": cat, **q, **info,
                                "rpm_range": [float(shaft.min() * 60), float(shaft.max() * 60)],
                                "sound": {k: (float(v) if v is not None else None) for k, v in sound.items()}})

    boost = 0.6 if has_drone else 1.6  # negatives carry more hard confusers
    for kind, prob in CONFUSER_PROBS.items():
        if rng.uniform() < min(1.0, prob * boost):
            cat, sig, t, p, level, cmeta = sample_confuser(rng, kind, duration, n_src)
            add(cat, ac.set_level_dba(sig, FS, level), t, p)
            meta["sources"].append({"category": cat, "level": level, **cmeta})

    # diffuse beds as plane waves from random directions, then wind and self-noise per mic
    pool = HELDOUT_BEDS + TRAIN_BEDS if heldout else TRAIN_BEDS
    kinds = list(rng.choice(pool, size=int(rng.integers(1, 3)), replace=False))
    if heldout and "rain" not in kinds:
        kinds[0] = "rain"
    bed_level = rng.uniform(45, 72) if heldout else rng.uniform(35, 65)
    bed = np.zeros((len(mic_pos), n_out), np.float32)
    for kind in kinds:
        for _ in range(6):
            x = src.bed(rng, kind, n_out, FS)
            az, el = rng.uniform(0, 2 * np.pi), rng.uniform(0, 0.4)
            u = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
            for j, m in enumerate(mic_pos):
                bed[j] += ac.fractional_shift(x, -(u @ m) / ac.C_SOUND, FS)
    bed *= ac.P_REF * 10 ** (bed_level / 20) / max(ac.a_weighted_rms(bed[0], FS), 1e-20)
    wind_level = rng.uniform(30, 70) if rng.uniform() < 0.6 else None
    for j in range(len(mic_pos)):
        if wind_level:
            bed[j] += ac.set_level_dba(src.mic_wind(rng, n_out, FS), FS, wind_level)
        bed[j] += ac.set_level_dba(rng.standard_normal(n_out).astype(np.float32), FS, rng.uniform(20, 32))
    per_cat["bed"] = bed
    meta.update(beds=kinds, bed_level=bed_level, wind_level=wind_level, duration=duration)

    audio = sum(per_cat.values())
    labels = frame_labels(per_cat, drone_tracks, mic_pos[0], n_out)
    return audio.astype(np.float32), labels, meta


def separate_regions(local_db, group=4, k=4):
    """Per frame, the k strongest tone regions that are genuinely separate.

    Bins are grouped (4 bins = 125 Hz); after taking the strongest group, it and its neighbours are
    suppressed before the next pick, so one tone straddling a group edge counts once. -> (frames, k)
    """
    g = local_db[: local_db.shape[0] // group * group].reshape(-1, group, local_db.shape[1]).max(1)
    frames = np.arange(g.shape[1])
    picks = []
    for _ in range(k):
        j = g.argmax(0)
        picks.append(g[j, frames].copy())
        for off in (-1, 0, 1):
            g[np.clip(j + off, 0, len(g) - 1), frames] = -np.inf
    return np.stack(picks, 1)


def frame_labels(per_cat, drone_tracks, mic, n_out, band=(50, 8000)):
    """Per 10 ms frame, for each category vs everything else (mic 0):

    snr_band/<cat>   broadband SNR over `band` (dB)
    snr_peak/<cat>   best single smoothed bin (dB)
    snr_lines/<cat>  (frames, 4): how far the k-th strongest separate tone region stands above the
                     rest (dB); see separate_regions
    bands/<cat>      coarse spectrum, 48 log bands 40 Hz-14 kHz at 25 frames/s (dB re 1 Pa^2), for relabelling
    """
    nfft = 1024
    power = {}
    for cat, x in per_cat.items():
        f, t, X = signal.stft(x[0], FS, nperseg=nfft, noverlap=nfft - HOP, boundary="even")
        power[cat] = np.abs(X) ** 2
    sel = (f >= band[0]) & (f <= band[1])
    total = sum(power.values())
    edges = np.geomspace(40, 14000, 49)
    band_idx = np.digitize(f, edges) - 1
    out = {"frame_t": t.astype(np.float32)}
    for cat, pc in power.items():
        coarse = np.stack([pc[band_idx == b].sum(0) for b in range(48)])
        n4 = coarse.shape[1] // 4 * 4
        coarse = coarse[:, :n4].reshape(48, -1, 4).mean(2)
        out[f"bands/{cat}"] = np.clip(10 * np.log10(coarse + 1e-20), -150, 50).astype(np.float16)
        if cat == "bed":
            continue
        rest = np.maximum(total - pc, 1e-30)
        out[f"snr_band/{cat}"] = np.clip(10 * np.log10(pc[sel].sum(0) / rest[sel].sum(0) + 1e-30), -100, 100).astype(np.float16)
        local = 10 * np.log10(np.maximum(uniform_filter(pc[sel], (3, 5)), 1e-30) / uniform_filter(rest[sel], (3, 5)))
        out[f"snr_peak/{cat}"] = np.clip(local.max(0), -100, 100).astype(np.float16)
        out[f"snr_lines/{cat}"] = np.clip(separate_regions(local), -100, 100).astype(np.float16)
    for i, (q, tc, p, shaft) in enumerate(drone_tracks):
        tau, r = ac.emission_time(t, tc, p, mic)
        dtau = np.gradient(tau, t)
        bpf = q["blades"] * np.interp(tau, tc, shaft.mean(0))
        pe = ac.interp_path(tc, p, tau)
        out[f"drone{i}/r"] = r.astype(np.float32)
        out[f"drone{i}/elev_deg"] = np.degrees(np.arcsin(np.clip((pe[:, 2] - mic[2]) / r, -1, 1))).astype(np.float32)
        out[f"drone{i}/rdot"] = np.gradient(r, t).astype(np.float32)
        out[f"drone{i}/bpf_emit"] = bpf.astype(np.float32)
        out[f"drone{i}/bpf_rx"] = (bpf * dtau).astype(np.float32)
    return out
