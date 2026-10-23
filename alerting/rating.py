"""Drone-likelihood rating (design.md 5.12, D-32) with coarse height (D-38).

    likelihood = p_cnn x f_height x f_speed x f_persist          each factor in [0, 1]

Height and speed come from the Kalman track (two or more flat clusters, D-30). Without a usable height
(no track, or height too uncertain) the alert is "unrated": it is never called a ground source on
missing evidence (D-38).

Levels:
  likely      likelihood >= LIKELY                          (take-cover / evacuate tiers, 13.4)
  possible    CNN fired but no rating, or LOW <= likelihood < LIKELY
  ground      likelihood < LOW with a usable height: probably a ground engine or other sound
              (logged at low priority, never discarded)
"""
from dataclasses import dataclass, field

import numpy as np

LIKELY, LOW = 0.5, 0.15
H_MID, H_WIDTH = 8.0, 2.5         # f_height = 0.5 at 8 m; ~0.06 on the ground, ~0.94 at 15 m (D-38)
SIGMA_H_MAX = 10.0                # m: a height less certain than this is not used
HOVER_H = 15.0                    # m: a slow track this high is a hovering drone, not a ground source
SPEED_DRONE = (2.0, 45.0)         # m/s: moving like a drone; faster is an aircraft, slower is walking pace
PERSIST_S = 2.0                   # s: time constant of f_persist


@dataclass
class Evidence:
    """What the rating sees for one alert (a track, or a CNN detection without a track)."""
    p_cnn: float                              # recent drone probability (max of the drone heads)
    detected_s: float                         # seconds the detection has been continuous
    duty: float = 1.0                         # fraction of the last 5 s with the CNN above threshold
    height: float | None = None               # m above the clusters (EKF)
    sigma_h: float | None = None              # m (EKF covariance)
    speed: float | None = None                # m/s horizontal+vertical (EKF)
    n_clusters: int = 0                       # clusters hearing it


@dataclass
class Rating:
    likelihood: float | None
    level: str
    factors: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)


def f_height(h):
    return float(1 / (1 + np.exp(-(h - H_MID) / H_WIDTH)))


def f_speed(speed, airborne):
    lo, hi = SPEED_DRONE
    if speed < lo:
        return 0.9 if airborne else 0.3       # hovering drone vs stationary/walking ground source
    if speed > hi:
        return 0.3                            # light aircraft and helicopters cruise faster
    return 1.0


def f_persist(detected_s, duty):
    return float((1 - np.exp(-detected_s / PERSIST_S)) * np.clip(duty, 0, 1))


def rate(e: Evidence) -> Rating:
    fp = f_persist(e.detected_s, e.duty)
    if e.height is None or e.sigma_h is None or e.sigma_h > SIGMA_H_MAX or e.speed is None:
        why = "no position yet" if e.height is None else f"height too uncertain (sigma {e.sigma_h:.0f} m)"
        return Rating(None, "possible", {"p_cnn": e.p_cnn, "persist": fp}, [why])
    fh = f_height(e.height)
    fs = f_speed(e.speed, airborne=e.height >= HOVER_H)
    like = float(e.p_cnn * fh * fs * fp)
    level = "likely" if like >= LIKELY else ("ground" if like < LOW else "possible")
    reasons = []
    reasons.append(f"height {e.height:.0f} m" + (" (airborne)" if fh > 0.8 else " (near the ground)" if fh < 0.2 else ""))
    reasons.append(f"speed {e.speed:.1f} m/s" + (" (hovering)" if e.speed < SPEED_DRONE[0] and e.height >= HOVER_H else ""))
    reasons.append(f"heard for {e.detected_s:.0f} s by {e.n_clusters} cluster(s)")
    return Rating(like, level, {"p_cnn": e.p_cnn, "height": fh, "speed": fs, "persist": fp}, reasons)
