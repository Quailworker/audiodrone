"""Multi-cluster tracker: extended Kalman filter on per-cluster directions (design.md 13.2; task step 3).

State s = [x, y, z, vx, vy, vz, ln f0, d ln f0/dt], constant velocity with white-noise acceleration q_acc;
ln f0 (the emitted pitch of the tracked harmonic family, loc/freqtrack.py) has a rate driven by white
noise q_f (D-13). The rate matters: clusters hear sound emitted up to ~0.3 s apart, and an RPM change over
that time otherwise looks like vertical motion (tested: v_z error 1-3 m/s without it).

Measurement from cluster k: the direction u of the sound that arrives now. It left the drone r/c ago,
so the model is evaluated at the emission position p_e = p - v * r / c (F7):

    u = (p_e - c_k) / |p_e - c_k|,   du/dp = (I - u u^T) / r,   du/dv = -(r/c) du/dp

The residual is taken in the tangent plane of the predicted direction, along e_az (horizontal) and e_el
(up), with separate noise sigma_az and sigma_el, so the flat cluster's weak elevation (design 7.8) can be
modelled: sigma_el = sigma_u / max(sin(el), sin_floor) for flat clusters, sigma_u for tetrahedra.

Doppler (design 13.2, D-16): a cluster whose direction updated a track can also give the pitch f of
its tracked tone(s). Model, at emission time (lag = r/c) and direction u:
ln f = ln f0 - lag * d ln f0/dt - ln(1 + u.v / c), with

    d/d ln f0 = 1,   d/d(rate) = -lag,   d/dv = -u / (c + rdot),   d/dp = -(I - u u^T) v / (r (c + rdot))

Clusters may lock onto different members of the harmonic family, so f is divided by the ratio in
freqtrack.RATIOS that brings it closest to the prediction; the pitch is used only if it then passes a
chi-square gate (1 dof). The first pitch seen initialises ln f0. Without `freqs` the filter is the
directions-only filter of v0.9 (ln f0 just carried along).

Association: each cluster gives up to two candidate directions per frame. Global nearest neighbour on
the Mahalanobis distance with a chi-square gate (2 dof). Unassigned candidates from >= 2 clusters in the
same frame start a track from the closest point between two rays. Tracks are dropped after `max_miss_s`
without an update.

Ground: flat at z = 0 by default. Over terrain (terra/), `ground(x, y)` gives the ground height in the
same frame; births below it are refused and tracks are kept above it. Births must lie within `birth_range`
of the centroid of the clusters.
"""
import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment

from loc.freqtrack import RATIOS

C = 343.0
GATE_CHI2 = 13.8                  # chi-square, 2 dof, 99.9 %
GATE_F_CHI2 = 9.0                 # chi-square, 1 dof (3 sigma), for the pitch
NX = 8


class Track:
    _next = 0

    def __init__(self, p, t, pos_sd=20.0, vel_sd=15.0):
        self.s = np.r_[p, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.P = np.diag([pos_sd ** 2] * 3 + [vel_sd ** 2] * 3 + [1.0, 0.05 ** 2])
        self.has_f = False                # ln f0 initialised from a pitch yet?
        self.n_f = 0                      # pitch updates
        self.t_last = t
        self.n_upd = 0
        self.id = Track._next
        Track._next += 1


class Tracker:
    def __init__(self, centres, flat, dt=0.1, q_acc=6.0, sigma_u=0.02, sin_floor=0.05, max_miss_s=3.0,
                 gate=GATE_CHI2, emission_correction=True, ground=None, birth_range=1500.0,
                 q_f=1e-3, sigma_lnf=0.003):
        self.c = np.asarray(centres, float)
        self.ground = ground or (lambda x, y: 0.0)
        self.birth_range = birth_range
        self.c_mid = self.c[:, :2].mean(0) if ground is not None else np.zeros(2)
        self.flat = flat
        self.dt, self.q_acc, self.sigma_u = dt, q_acc, sigma_u
        self.sin_floor, self.max_miss, self.gate = sin_floor, max_miss_s, gate
        self.emission = emission_correction
        self.q_f, self.sigma_lnf = q_f, sigma_lnf
        F = np.eye(NX)
        F[:3, 3:6] = dt * np.eye(3)
        F[6, 7] = dt
        self.F = F
        q = q_acc * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        Q = np.zeros((NX, NX))
        for i in range(3):
            Q[np.ix_([i, i + 3], [i, i + 3])] = q
        Q[6:, 6:] = q_f * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])      # pitch-rate noise /s^3
        self.Q = Q
        self.tracks = []

    # --- model ---------------------------------------------------------------------------
    def predict_meas(self, s, k):
        """Predicted direction from cluster k, its tangent basis (2, 3) and Jacobian (2, 6)."""
        p, v = s[:3], s[3:6]
        d0 = p - self.c[k]
        r0 = np.linalg.norm(d0)
        lag = r0 / C if self.emission else 0.0
        d = d0 - v * lag
        r = max(np.linalg.norm(d), 1e-3)
        u = d / r
        e_az = np.array([-u[1], u[0], 0.0])
        n = np.linalg.norm(e_az)
        e_az = e_az / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        e_el = np.cross(u, e_az)
        if e_el[2] < 0:
            e_el = -e_el
        B = np.vstack([e_az, e_el])
        Ju = (np.eye(3) - np.outer(u, u)) / r
        H = np.hstack([B @ Ju, -lag * (B @ Ju), np.zeros((2, 2))])
        return u, B, H

    def predict_pitch(self, s, k):
        """Predicted ln f at cluster k and its Jacobian (NX,)."""
        p, v = s[:3], s[3:6]
        d0 = p - self.c[k]
        lag = np.linalg.norm(d0) / C if self.emission else 0.0
        d = d0 - v * lag
        r = max(np.linalg.norm(d), 1e-3)
        u = d / r
        rdot = float(u @ v)
        a = 1.0 / (C + rdot)
        H = np.zeros(NX)
        H[:3] = -a * ((np.eye(3) - np.outer(u, u)) @ v) / r
        H[3:6] = -a * u
        H[6] = 1.0
        H[7] = -lag
        return s[6] - lag * s[7] - np.log1p(rdot / C), H

    def pitch_update(self, tr, k, freqs):
        """Doppler update of track tr from cluster k's tone pitches (Hz). Returns True if one was used."""
        g, H = self.predict_pitch(tr.s, k)
        R = self.sigma_lnf ** 2
        if not tr.has_f:                                   # first pitch: set ln f0, keep the kinematics
            f = freqs[0]
            tr.s[6] = np.log(f) + (tr.s[6] - g)            # ln f + ln(1 + rdot/c)
            tr.P[6:, :] = 0.0
            tr.P[:, 6:] = 0.0
            tr.P[6, 6], tr.P[7, 7] = 0.05 ** 2, 0.05 ** 2
            tr.has_f = True
            return True
        S = float(H @ tr.P @ H) + R
        best = None
        for f in freqs:
            y = np.log(f / RATIOS) - g                     # every family member the tone could be
            j = int(np.argmin(np.abs(y)))
            d2 = y[j] ** 2 / S
            if d2 < GATE_F_CHI2 and (best is None or d2 < best[0]):
                best = (d2, y[j])
        if best is None:
            return False
        K = tr.P @ H / S
        tr.s = tr.s + K * best[1]
        IKH = np.eye(NX) - np.outer(K, H)
        tr.P = IKH @ tr.P @ IKH.T + R * np.outer(K, K)
        tr.n_f += 1
        return True

    def meas_cov(self, u, k):
        sd_el = self.sigma_u / max(abs(u[2]), self.sin_floor) if self.flat else self.sigma_u
        return np.diag([self.sigma_u ** 2, sd_el ** 2])

    def innovation(self, tr, k, z):
        u, B, H = self.predict_meas(tr.s, k)
        y = B @ (z - u)
        R = self.meas_cov(u, k)
        return y, H @ tr.P @ H.T + R, H, R

    # --- one frame ------------------------------------------------------------------------
    def step(self, t, meas, freqs=None):
        """meas: list of (cluster k, unit vector z); freqs: optional {cluster k: [tone pitch, Hz]}.
        Returns the tracks alive after this frame."""
        for tr in self.tracks:
            tr.s = self.F @ tr.s
            tr.P = self.F @ tr.P @ self.F.T + self.Q
        used = set()
        if self.tracks and meas:
            cost = np.full((len(self.tracks), len(meas)), 1e6)
            for i, tr in enumerate(self.tracks):
                for j, (k, z) in enumerate(meas):
                    y, S, _, _ = self.innovation(tr, k, z)
                    d2 = float(y @ np.linalg.solve(S, y))
                    if d2 < self.gate:
                        cost[i, j] = d2
            # one measurement per track per cluster: assign cluster by cluster
            for k in sorted({k for k, _ in meas}):
                cols = [j for j, (kk, _) in enumerate(meas) if kk == k]
                rows, cs = linear_sum_assignment(cost[:, cols])
                for i, cj in zip(rows, cs, strict=True):
                    j = cols[cj]
                    if cost[i, j] < 1e6:
                        self.update(self.tracks[i], *meas[j])
                        self.tracks[i].t_last = t
                        used.add(j)
                        if freqs and freqs.get(k):
                            self.pitch_update(self.tracks[i], k, freqs[k])
        free = [m for j, m in enumerate(meas) if j not in used]
        self.birth(t, free)
        self.tracks = [tr for tr in self.tracks if t - tr.t_last <= self.max_miss]
        self.merge()
        for tr in self.tracks:
            tr.s[2] = max(tr.s[2], self.ground(tr.s[0], tr.s[1]))     # nothing flies below the ground
        return self.tracks

    def merge(self, dist=10.0):
        """Drop the younger of two tracks that have converged on the same drone."""
        keep = []
        for tr in sorted(self.tracks, key=lambda x: -x.n_upd):
            if all(np.linalg.norm(tr.s[:3] - o.s[:3]) > dist for o in keep):
                keep.append(tr)
        self.tracks = keep

    def update(self, tr, k, z):
        y, S, H, R = self.innovation(tr, k, z)
        K = tr.P @ H.T @ np.linalg.inv(S)
        tr.s = tr.s + K @ y
        IKH = np.eye(NX) - K @ H
        tr.P = IKH @ tr.P @ IKH.T + K @ R @ K.T                        # Joseph form, same R as in S
        tr.n_upd += 1

    def birth(self, t, free):
        """Start one track from the best pair of rays from two different clusters, if they nearly meet."""
        best = None
        for (k1, z1), (k2, z2) in itertools.combinations(free, 2):
            if k1 == k2:
                continue
            p, gap, ang = closest_point(self.c[k1], z1, self.c[k2], z2)
            if p is None or p[2] < self.ground(p[0], p[1]) or np.linalg.norm(p[:2] - self.c_mid) > self.birth_range:
                continue
            rng = min(np.linalg.norm(p - self.c[k1]), np.linalg.norm(p - self.c[k2]))
            if gap > 3.0 + 0.05 * rng or ang < np.radians(5):
                continue
            score = ang - gap / 100
            if best is None or score > best[0]:
                best = (score, p)
        if best is not None:
            tr = Track(best[1], t)
            self.tracks.append(tr)


def closest_point(c1, u1, c2, u2):
    """Midpoint of the shortest segment between rays c1 + a u1 and c2 + b u2 (a, b > 0).
    Returns (point, gap length, angle between rays) or (None, inf, 0) if they diverge."""
    w = c1 - c2
    b = u1 @ u2
    den = 1 - b * b
    if den < 1e-6:
        return None, np.inf, 0.0
    d, e = u1 @ w, u2 @ w
    a1 = (b * e - d) / den
    a2 = (e - b * d) / den
    if a1 <= 0 or a2 <= 0:
        return None, np.inf, 0.0
    q1, q2 = c1 + a1 * u1, c2 + a2 * u2
    return (q1 + q2) / 2, float(np.linalg.norm(q1 - q2)), float(np.arccos(np.clip(b, -1, 1)))


def run_tracker(times, meas_per_frame, centres, flat, freqs_per_frame=None, **kw):
    """meas_per_frame: list over frames of [(k, u)]; freqs_per_frame: optional list over frames of
    {k: [pitch Hz]} (Doppler). Returns per frame a list of (track id, state (8,), P diagonal,
    number of updates, seconds since last update)."""
    Track._next = 0
    trk = Tracker(centres, flat, **kw)
    out = []
    freqs_per_frame = freqs_per_frame or [None] * len(times)
    for t, meas, fq in zip(times, meas_per_frame, freqs_per_frame, strict=True):
        tracks = trk.step(t, meas, fq)
        out.append([(tr.id, tr.s.copy(), np.diag(tr.P).copy(), tr.n_upd, t - tr.t_last) for tr in tracks])
    return out
