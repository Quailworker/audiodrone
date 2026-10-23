"""Danger-zone prediction (design.md 13.3, D-35).

From a Kalman track (position p, velocity v, covariance P) and a protected site s with danger radius R,
in the horizontal plane (sites are on the ground; height is reported, not used for the geometry):

    predicted position   m(tau) = p + v tau,  spread sigma(tau) from the constant-velocity prediction of P
    closest approach     tau* = -(dp.v) / |v|^2,  d_min = |dp + v tau*|            (dp = p - s)
    probability inside   q(tau) = P(|X(tau) - s| <= R),  X(tau) ~ N(m(tau), sigma(tau)^2 I)
                               = ncx2.cdf((R / sigma)^2, df=2, nc=(|m(tau) - s| / sigma)^2)
    threatened           max over 0 <= tau <= T_ALERT of q(tau) >= Q_MIN          (automatic take-cover)
    arrival              first tau with q(tau) >= Q_MIN
    on course            the predicted (mean) path passes within R of the site within T_ALERT, whatever q is
                         (early warning to operators; any evacuation from it needs their approval)

The spread uses a prediction model for purposeful flight (Q_PRED), not the tracker's process noise:
the tracker's q_acc = 6 is tuned to follow manoeuvres between measurements and makes 20 s predictions
+-170 m wide, so a drone flying straight at a site from 300 m scored an 8 % chance and raised no alert.

v0.9.1 change: the first version widened the radius to R + k sigma(tau*) and applied it from now on.
For slow drones tau* is far ahead and sigma grows ~ tau^1.5 under manoeuvre noise, so the widened radius
reached hundreds of metres and a drone 200 m away read as "over the site now". The probability form
spreads an uncertain prediction thin instead of making the zone huge (alerting/replay.py, scene L20000014).
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import ncx2

T_ALERT = 120.0                   # s: look-ahead for alerts (open question 34)
STEP = 0.5                        # s: look-ahead grid
Q_MIN = 0.2                       # probability of being inside R that counts as a threat
Q_PRED = 0.5                      # (m/s^2)^2/s: manoeuvre noise for prediction (purposeful flight)
K_SIGMA = 2.0                     # for the drawn corridor and the people search radius only


@dataclass
class Threat:
    site: str
    distance_now: float           # m, horizontal
    d_min: float                  # m, closest approach of the predicted path
    tau_min: float                # s until closest approach (0 if moving away)
    tau_arrive: float | None      # s until q >= Q_MIN (None: not threatened within T_ALERT)
    q_max: float                  # highest probability of being inside R within T_ALERT
    sigma_at_arrival: float       # m, position spread when the threat starts (or at closest approach)
    threatened: bool
    on_course: bool = False       # mean path passes within R within T_ALERT


def sigma_at(P, tau, q_acc=Q_PRED):
    """Horizontal position standard deviation tau seconds ahead (per axis, averaged over x and y)."""
    tau = np.asarray(tau, float)
    var = [P[i, i] + 2 * tau * P[i, i + 3] + tau ** 2 * P[i + 3, i + 3] + q_acc * tau ** 3 / 3 for i in (0, 1)]
    return np.sqrt(np.maximum((var[0] + var[1]) / 2, 1e-6))


def prob_inside(dist, sigma, radius):
    """P(|X| <= radius) for a 2-D isotropic Gaussian with mean at distance `dist` and per-axis `sigma`."""
    return ncx2.cdf((radius / sigma) ** 2, df=2, nc=(np.asarray(dist) / sigma) ** 2)


def assess(state, P, site_xy, radius, name="site", t_alert=T_ALERT, q_min=Q_MIN):
    p, v = np.asarray(state[:2], float), np.asarray(state[3:5], float)
    s = np.asarray(site_xy, float)
    dp = p - s
    vv = float(v @ v)
    tau_min = max(0.0, -float(dp @ v) / vv) if vv > 1e-6 else 0.0
    taus = np.arange(0.0, t_alert + 1e-9, STEP)
    dist = np.linalg.norm(dp[None] + v[None] * taus[:, None], axis=1)
    sig = sigma_at(P, taus)
    q = prob_inside(dist, sig, radius)
    hit = np.flatnonzero(q >= q_min)
    tau_arr = float(taus[hit[0]]) if len(hit) else None
    sig_ref = float(sig[hit[0]]) if len(hit) else float(sigma_at(P, tau_min))
    d_min = float(np.linalg.norm(dp + v * tau_min))
    return Threat(name, float(np.linalg.norm(dp)), d_min, tau_min, tau_arr, float(q.max()), sig_ref,
                  tau_arr is not None, bool(d_min <= radius and tau_min <= t_alert))


def corridor(state, P, radius, t_max=60.0, step=5.0, k=K_SIGMA):
    """Predicted path points and half-widths [(t, x, y, half_width_m)] for drawing the danger zone."""
    p, v = np.asarray(state[:2], float), np.asarray(state[3:5], float)
    return [(float(t), *map(float, p + v * t), float(radius + k * sigma_at(P, t)))
            for t in np.arange(0, t_max + 1e-9, step)]
