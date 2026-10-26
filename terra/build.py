"""Build mode helpers (Border watch): how far a cluster hears each drone type, and how long a run will take.

Listening range: the distance at which a typical drone of the class (median level from sim/scene.py
sample_drone, dB(A) at 1 m) falls to MARGIN_DB below the ambient bed, after spherical spreading and ISO 9613-1
air absorption at F_TONE. The tonal lines a cluster listens to stand out of a broadband bed, so a drone is
heard below the bed's A-level. MARGIN_DB = 14 is fitted to what the full simulation gave:
  - desert (35 dB(A)): the 7-12 in quad first heard at about 0.9-1.2 km, the FPV at 1.15-1.45 km (results/spacing);
  - town (52 dB(A)): the two-stroke multirotor tracked from 506 m, the quads unheard at 200-400 m
    (results/swarm_abudhabi).
It is a guide on open ground: hills and buildings that block the sound only show in a full run.
To locate a drone at least two clusters must hear it, so the advice is "no more than one range apart".
"""
import math
from functools import lru_cache

import numpy as np

from sim.acoustics import air_absorption_db_per_m

from .scenario import AMBIENT, DRONE_CLASSES

MARGIN_DB = 14.0
F_TONE = 500.0
LABELS = {"fpv": "FPV racer (5-7 in)", "mid": "Quadcopter (7-12 in)", "ts_mr": "Large multirotor (petrol)",
          "ts_fw": "Fixed-wing (petrol)"}
AMBIENT_LABELS = {"desert": "Quiet desert", "desert_wind": "Windy desert", "town": "Town", "highway": "Highway"}
SECONDS_PER_CLUSTER_DRONE_MINUTE = 4.5          # fitted on this desktop (3 render workers): see estimate_s


@lru_cache(maxsize=1)
def class_levels():
    from sim.scene import sample_drone
    return {c: float(np.median([sample_drone(np.random.default_rng(s), c, False)["level"] for s in range(300)]))
            for c in DRONE_CLASSES}


def listening_range_m(cls, ambient, temp_c=35.0, rh=30.0):
    """Distance (m) at which a typical drone of `cls` is MARGIN_DB under the `ambient` bed."""
    level = class_levels()[cls]
    target = AMBIENT[ambient][1] - MARGIN_DB
    alpha = float(air_absorption_db_per_m(F_TONE, temp_c, rh))
    lo, hi = 1.0, 20000.0
    for _ in range(60):                                      # bisection on L1 - 20 log r - alpha r = target
        mid = math.sqrt(lo * hi)
        if level - 20 * math.log10(mid) - alpha * mid >= target:
            lo = mid
        else:
            hi = mid
    return round(lo / 10) * 10


def ranges(temp_c=35.0, rh=30.0):
    """{ambient: {class: range m}} plus labels, for the screen."""
    return {"ranges": {a: {c: listening_range_m(c, a, temp_c, rh) for c in DRONE_CLASSES} for a in AMBIENT},
            "classes": LABELS, "ambients": AMBIENT_LABELS, "levels": class_levels(), "margin_db": MARGIN_DB}


def flight_s(drone):
    wp = np.asarray(drone["waypoints"], float)
    return float(np.linalg.norm(np.diff(wp, axis=0), axis=1).sum() / max(float(drone.get("speed", 12.0)), 0.5)) + 5.0


def estimate_s(scn):
    """Rough run time: rendering dominates and grows with clusters x drones x flight length."""
    if not scn.get("drones") or not scn.get("clusters"):
        return 0
    minutes = min(max(flight_s(d) for d in scn["drones"]), 300.0) / 60.0
    return round(20 + SECONDS_PER_CLUSTER_DRONE_MINUTE * len(scn["clusters"]) * len(scn["drones"]) * minutes)
