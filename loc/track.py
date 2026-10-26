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


# Tracker profile for flat ground clusters (design v0.12-v0.13; terra/pipeline.py uses it). The bare Tracker
# defaults stay as in v0.9 so the Phase 1 results reproduce; any chain opts in with Tracker(..., **TRACKER_V12).
#   birth_rays=3, confirm_deg=4      a new track needs a third cluster's bearing to agree (when 3+ clusters hear)
#   max_agl_m=800                    nothing we could hear flies higher; rays crossing up there are ghosts
#   merge_h_m=150, birth_excl_m=150  one direction per cluster: two tracks within 150 m in plan are one drone
#   merge_ang_deg=4, merge_floor_m=40  several directions per cluster: merge only what no cluster can tell apart
TRACKER_V12 = {"birth_rays": 3, "confirm_deg": 4.0, "max_agl_m": 800.0, "merge_h_m": 150.0, "merge_dv": 25.0,
               "birth_excl_m": 150.0, "merge_ang_deg": 4.0, "merge_floor_m": 40.0}


class Track:
    _next = 0

    def __init__(self, p, t, pos_sd=20.0, vel_sd=15.0):
        self.s = np.r_[p, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.P = np.diag([pos_sd ** 2] * 3 + [vel_sd ** 2] * 3 + [1.0, 0.05 ** 2])
        self.has_f = False                # ln f0 initialised from a pitch yet?
        self.heard_by = {}                # cluster -> last time it updated this track
        self.n_f = 0                      # pitch updates
        self.t_last = t
        self.n_upd = 0
        self.id = Track._next
        Track._next += 1


class Tracker:
    def __init__(self, centres, flat, dt=0.1, q_acc=6.0, sigma_u=0.02, sin_floor=0.05, max_miss_s=3.0,
                 gate=GATE_CHI2, emission_correction=True, ground=None, birth_range=1500.0,
                 q_f=1e-3, sigma_lnf=0.003, merge_h_m=None, birth_excl_m=None, merge_dv=6.0,
                 max_agl_m=None, min_support=None, support_s=2.0, max_speed=None, max_vz=None,
                 birth_rays=2, confirm_deg=6.0, merge_ang_deg=None, merge_floor_m=40.0, merge_k=3.0):
        # Flat ground clusters fix height poorly, so tracks on one drone can agree in plan but differ by
        # hundreds of metres in height. merge_h_m / birth_excl_m (None = off, the v0.11 behaviour) merge
        # tracks and refuse births by horizontal distance, ignoring height: see merge() and birth().
        self.merge_h_m, self.birth_excl_m, self.merge_dv = merge_h_m, birth_excl_m, merge_dv
        # Plausibility (None = off): bad directions from far clusters cross in odd places and start tracks
        # kilometres up or away (ghosts). A real target here flies below max_agl_m above the ground, keeps
        # being heard by min_support different clusters within support_s, and moves no faster than
        # max_speed horizontally / max_vz vertically. See prune().
        self.max_agl, self.min_support, self.support_s = max_agl_m, min_support, support_s
        self.max_speed, self.max_vz = max_speed, max_vz
        # Births: with flat clusters any two bearings cross somewhere, so a bad bearing plus a good one starts a
        # ghost. birth_rays=3 asks a third cluster's free direction to point at the candidate within
        # confirm_deg (in azimuth) before a track starts, whenever 3 or more clusters gave free directions.
        self.birth_rays, self.confirm = birth_rays, np.radians(confirm_deg)
        # Swarm-safe duplicate handling (merge_ang_deg set; replaces the fixed merge_h_m / birth_excl_m):
        # two tracks are one drone if they are within max(merge_floor_m, merge_k x their combined horizontal sd)
        # or if no cluster that recently heard them sees them more than merge_ang_deg apart in azimuth (the
        # data cannot tell them apart); and a birth is refused when both its rays point at an existing track
        # (they are that track's bearings). Two real drones that some cluster sees apart stay two tracks.
        self.merge_ang = None if merge_ang_deg is None else np.radians(merge_ang_deg)
        self.merge_floor, self.merge_k = merge_floor_m, merge_k
        # Which rule applies depends on the front end. With one direction per cluster per frame, a cluster can
        # only ever feed one of two close tracks, so a second track near the first is always a duplicate and
        # the fixed merge_h_m / birth_excl_m rule is right (two drones that close cannot be told apart). Once
        # clusters report several directions (one per tone line), the swarm-safe rule above is used instead.
        # multi_recent is the last time any cluster gave 2+ directions in one frame.
        self.multi_recent = -np.inf
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
        per_cluster = {}
        for k, _ in meas:
            per_cluster[k] = per_cluster.get(k, 0) + 1
        if any(n >= 2 for n in per_cluster.values()):
            self.multi_recent = t                                   # the front end reports several directions
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
                        self.tracks[i].heard_by[k] = t
                        used.add(j)
                        if freqs and freqs.get(k):
                            self.pitch_update(self.tracks[i], k, freqs[k])
        free = [m for j, m in enumerate(meas) if j not in used]
        self.birth(t, free)
        self.tracks = [tr for tr in self.tracks if t - tr.t_last <= self.max_miss]
        self.prune(t)
        self.merge()
        for tr in self.tracks:
            tr.s[2] = max(tr.s[2], self.ground(tr.s[0], tr.s[1]))     # nothing flies below the ground
        return self.tracks

    def merge(self, dist=10.0):
        """Drop the younger of two tracks that have converged on the same drone: within `dist` in 3D, or, with
        merge_h_m set, within max(merge_h_m, 2 x their combined horizontal sd) in plan and moving alike
        (horizontal velocities within merge_dv), whatever their heights."""
        keep = []
        for tr in sorted(self.tracks, key=lambda x: -x.n_upd):
            if all(not self._same(tr, o, dist) for o in keep):
                keep.append(tr)
        self.tracks = keep

    def prune(self, t):
        """Drop implausible tracks: too high above the ground, too fast, or (once confirmed) no longer heard
        by min_support different clusters within support_s."""
        keep = []
        for tr in self.tracks:
            x, y, z = tr.s[:3]
            if self.max_agl is not None and z - self.ground(x, y) > self.max_agl:
                continue
            if self.max_speed is not None and np.hypot(tr.s[3], tr.s[4]) > self.max_speed:
                continue
            if self.max_vz is not None and abs(tr.s[5]) > self.max_vz:
                continue
            if (self.min_support is not None and tr.n_upd >= 5
                    and sum(1 for tk in tr.heard_by.values() if t - tk <= self.support_s) < self.min_support):
                continue
            keep.append(tr)
        self.tracks = keep

    def _swarm_rule(self, t):
        return self.merge_ang is not None and t - self.multi_recent <= self.support_s

    def _same(self, a, b, dist):
        if np.linalg.norm(a.s[:3] - b.s[:3]) <= dist:
            return True
        if self._swarm_rule(max(a.t_last, b.t_last)):
            return self._indistinct(a, b)
        if self.merge_h_m is None:
            return False
        sd = np.sqrt(max(a.P[0, 0], a.P[1, 1]) + max(b.P[0, 0], b.P[1, 1]))
        return (np.linalg.norm(a.s[:2] - b.s[:2]) <= max(self.merge_h_m, 2.0 * sd)
                and np.linalg.norm(a.s[3:5] - b.s[3:5]) <= self.merge_dv)

    def _az(self, k, p):
        d = p - self.c[k]
        return np.arctan2(d[0], d[1])

    def _indistinct(self, a, b):
        """True if a and b are one drone as far as the data can tell (see __init__, merge_ang_deg)."""
        if np.linalg.norm(a.s[3:5] - b.s[3:5]) > self.merge_dv:
            return False
        gap = np.linalg.norm(a.s[:2] - b.s[:2])
        sd = np.sqrt(max(a.P[0, 0], a.P[1, 1]) + max(b.P[0, 0], b.P[1, 1]))
        if gap <= max(self.merge_floor, self.merge_k * sd):
            return True
        t = max(a.t_last, b.t_last)
        ks = {k for tr in (a, b) for k, tk in tr.heard_by.items() if t - tk <= self.support_s}
        if len(ks) < 2:
            return False
        return all(abs((self._az(k, a.s[:3]) - self._az(k, b.s[:3]) + np.pi) % (2 * np.pi) - np.pi) <= self.merge_ang
                   for k in ks)

    def _explained(self, k1, z1, k2, z2):
        """True if both rays point at one existing track: they are its bearings, not a new drone."""
        for o in self.tracks:
            if all(abs((np.arctan2(z[0], z[1]) - self._az(k, o.s[:3]) + np.pi) % (2 * np.pi) - np.pi) <= self.confirm
                   for k, z in ((k1, z1), (k2, z2))):
                return True
        return False

    def update(self, tr, k, z):
        y, S, H, R = self.innovation(tr, k, z)
        K = tr.P @ H.T @ np.linalg.inv(S)
        tr.s = tr.s + K @ y
        IKH = np.eye(NX) - K @ H
        tr.P = IKH @ tr.P @ IKH.T + K @ R @ K.T                        # Joseph form, same R as in S
        tr.n_upd += 1

    def birth(self, t, free):
        """Start one track from the best pair of rays from two different clusters, if they nearly meet (and, with
        birth_rays=3 and at least 3 clusters giving free directions, if a third cluster points there too)."""
        best = None
        others = {}
        for k, z in free:
            others.setdefault(k, []).append(z)
        need3 = self.birth_rays >= 3 and len(others) >= 3
        for (k1, z1), (k2, z2) in itertools.combinations(free, 2):
            if k1 == k2:
                continue
            p, gap, ang = closest_point(self.c[k1], z1, self.c[k2], z2)
            if p is None or p[2] < self.ground(p[0], p[1]) or np.linalg.norm(p[:2] - self.c_mid) > self.birth_range:
                continue
            if self.max_agl is not None and p[2] - self.ground(p[0], p[1]) > self.max_agl:
                continue                    # rays crossing kilometres up: not a drone we could hear
            rng = min(np.linalg.norm(p - self.c[k1]), np.linalg.norm(p - self.c[k2]))
            if gap > 3.0 + 0.05 * rng or ang < np.radians(5):
                continue
            if self._swarm_rule(t):
                if self._explained(k1, z1, k2, z2):
                    continue                # both rays point at an existing track: its bearings, not a new drone
            elif self.birth_excl_m is not None and any(np.linalg.norm(p[:2] - o.s[:2]) <= self.birth_excl_m for o in self.tracks):
                continue                    # an existing track is already there in plan: a duplicate, not a new drone
            score = ang - gap / 100
            if need3:
                agree = sum(1 for k3 in others if k3 not in (k1, k2) and self._points_at(k3, others[k3], p))
                if agree == 0:
                    continue                # no third cluster backs this crossing: likely a ghost
                score += agree              # more clusters agreeing beats a tighter crossing
            if best is None or score > best[0]:
                best = (score, p, k1, k2)
        if best is not None:
            tr = Track(best[1], t)
            tr.heard_by = {best[2]: t, best[3]: t}
            self.tracks.append(tr)


    def _points_at(self, k, zs, p):
        """True if any of cluster k's directions zs points at p within confirm_deg in azimuth (flat clusters
        measure azimuth well and elevation poorly)."""
        d = p - self.c[k]
        az = np.arctan2(d[0], d[1])
        return any(abs((np.arctan2(z[0], z[1]) - az + np.pi) % (2 * np.pi) - np.pi) <= self.confirm for z in zs)


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
