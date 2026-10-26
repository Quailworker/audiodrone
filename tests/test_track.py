"""Tracker on perfect and noisy synthetic measurements (tasks/phase1_localisation.md step 3)."""
import numpy as np

from loc.track import closest_point, run_tracker
from sim.acoustics import emission_time, interp_path
from sim.layouts import field


def truth_pass(duration=40.0, alt=40.0, speed=12.0):
    t = np.arange(-10, duration + 1, 0.02)
    x = -240 + speed * (t + 0)
    p = np.stack([x, 0.3 * x + 20, alt + 5 * np.sin(t / 6)], 1)
    return t, p


def measurements(t_ctrl, p, centres, times, sigma=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for c in centres:
        tau, r = emission_time(times, t_ctrl, p, c)
        u = (interp_path(t_ctrl, p, tau) - c) / r[:, None]
        if sigma:
            u = u + rng.normal(0, sigma, u.shape)
            u /= np.linalg.norm(u, axis=1, keepdims=True)
        out.append(u)
    return [[(k, out[k][i]) for k in range(len(centres))] for i in range(len(times))]


def errors(res, t_ctrl, p, times, skip_s=5.0):
    e = []
    for t, tracks in zip(times, res, strict=True):
        if t < skip_s or not tracks:
            continue
        best = max(tracks, key=lambda x: x[3])
        e.append(np.linalg.norm(best[1][:3] - interp_path(t_ctrl, p, np.array([t]))[0]))
    return np.array(e)


def test_closest_point():
    p, gap, _ = closest_point(np.zeros(3), np.array([1.0, 0, 0]), np.array([10.0, -5, 0]), np.array([0, 1.0, 0]))
    assert np.allclose(p, [10, 0, 0]) and gap < 1e-9


def test_perfect_measurements_recover_track():
    centres, _ = field("square5_100", "tetra", "pole")
    t_ctrl, p = truth_pass()
    times = np.arange(0.05, 40, 0.1)
    res = run_tracker(times, measurements(t_ctrl, p, centres, times), centres, flat=False, sigma_u=0.002)
    e = errors(res, t_ctrl, p, times)
    assert len(e) > 0.85 * np.sum(times >= 5)
    assert np.median(e) < 0.5 and np.percentile(e, 95) < 1.5, (np.median(e), np.percentile(e, 95))


def test_noisy_measurements_flat():
    centres, _ = field("square5_100", "flat", "ground")
    t_ctrl, p = truth_pass()
    times = np.arange(0.05, 40, 0.1)
    res = run_tracker(times, measurements(t_ctrl, p, centres, times, sigma=0.01), centres, flat=True, sigma_u=0.01)
    e = errors(res, t_ctrl, p, times)
    assert np.median(e) < 3.0, np.median(e)


def test_emission_correction_matters():
    """Without the r/c correction a 12 m/s drone ~100 m away is tracked ~3 m behind (design 13.2)."""
    centres, _ = field("square5_100", "tetra", "pole")
    t_ctrl, p = truth_pass()
    times = np.arange(0.05, 40, 0.1)
    m = measurements(t_ctrl, p, centres, times)
    on = np.median(errors(run_tracker(times, m, centres, False, sigma_u=0.002), t_ctrl, p, times))
    off = np.median(errors(run_tracker(times, m, centres, False, sigma_u=0.002, emission_correction=False), t_ctrl, p, times))
    assert on < off / 2, (on, off)


def test_covariance_consistent():
    """Normalised estimation error squared (3 dof) near 3 means the reported covariance is honest.
    The filter is a little conservative (q_acc is generous for this smooth path): accept 0.8-6."""
    from loc.track import Track, Tracker
    centres, _ = field("square5_100", "tetra", "pole")
    t_ctrl, p = truth_pass()
    times = np.arange(0.05, 40, 0.1)
    nees = []
    for seed in range(3):
        Track._next = 0
        trk = Tracker(centres, False, sigma_u=0.01)
        for t, m in zip(times, measurements(t_ctrl, p, centres, times, sigma=0.01, seed=seed), strict=True):
            tracks = trk.step(t, m)
            if t > 8 and tracks:
                tr = max(tracks, key=lambda x: x.n_upd)
                e = tr.s[:3] - interp_path(t_ctrl, p, np.array([t]))[0]
                nees.append(e @ np.linalg.solve(tr.P[:3, :3], e))
    assert 0.8 < np.mean(nees) < 6.0, np.mean(nees)


def pitches(t_ctrl, p, centres, times, f0=180.0, sd=0.002, seed=0, octave_cluster=2):
    """Pitch heard at each cluster: f0(emission time) / (1 + rdot / c), RPM wander, 0.2 % noise.
    One cluster locks onto the second harmonic (alignment must undo it)."""
    rng = np.random.default_rng(seed)
    v = np.gradient(p, t_ctrl, axis=0)
    out = []
    for k, c in enumerate(centres):
        tau, r = emission_time(times, t_ctrl, p, c)
        u = (interp_path(t_ctrl, p, tau) - c) / r[:, None]
        rdot = np.einsum("ij,ij->i", u, interp_path(t_ctrl, v, tau))
        f = f0 * (1 + 0.03 * np.sin(tau / 4)) / (1 + rdot / 343.0) * np.exp(rng.normal(0, sd, len(times)))
        out.append(f * (2 if k == octave_cluster else 1))
    return [{k: [out[k][i]] for k in range(len(centres))} for i in range(len(times))]


def test_doppler_improves_velocity_flat():
    """Directions + pitch (D-16) vs directions only, flat clusters, 0.01 direction noise, 0.1 % pitch noise.
    Doppler about halves the horizontal velocity error; with every cluster on the ground it adds little
    to v_z (design 13.1), so v_z must only not get worse."""
    centres, _ = field("square5_100", "flat", "ground")
    t_ctrl, p = truth_pass()
    v = np.gradient(p, t_ctrl, axis=0)
    times = np.arange(0.05, 40, 0.1)
    m = measurements(t_ctrl, p, centres, times, sigma=0.01)
    rms = {}
    for name, fq in (("dir", None), ("dop", pitches(t_ctrl, p, centres, times, sd=0.001))):
        res = run_tracker(times, m, centres, True, freqs_per_frame=fq, sigma_u=0.01, sigma_lnf=0.002)
        e = np.array([max(tr, key=lambda x: x[3])[1][3:6] - interp_path(t_ctrl, v, np.array([t]))[0]
                      for t, tr in zip(times, res, strict=True) if t > 8 and tr])
        rms[name] = np.sqrt((e ** 2).mean(0))
    h = {k: np.hypot(*r[:2]) for k, r in rms.items()}
    assert h["dop"] < 0.65 * h["dir"], rms
    assert rms["dop"][2] < 1.3 * rms["dir"][2], rms


def test_tracks_agreeing_in_plan_merge_whatever_their_heights():
    """Flat clusters fix height poorly: two tracks over the same spot at 80 m and 400 m are one drone."""
    from loc.track import Track, Tracker
    trk = Tracker(np.zeros((3, 3)) + [[0, 0, 0]], flat=True, merge_h_m=150.0, merge_dv=25.0)
    a, b = Track(np.array([500.0, 500.0, 80.0]), 0.0), Track(np.array([590.0, 520.0, 400.0]), 0.0)
    a.n_upd, b.n_upd = 50, 10
    trk.tracks = [a, b]
    trk.merge()
    assert trk.tracks == [a]                                       # the older one is kept
    off = Tracker(np.zeros((3, 3)), flat=True)                     # without merge_h_m: 3D distance only
    off.tracks = [a, b]
    off.merge()
    assert len(off.tracks) == 2


def test_birth_needs_a_third_cluster_when_three_hear():
    """Any two flat-cluster bearings cross somewhere; a ghost from one bad bearing is refused when a third
    cluster is available to confirm, and the real drone is born where three agree."""
    from loc.track import Track, Tracker
    C = np.array([[0.0, 0, 0], [1000, 0, 0], [0, 1000, 0], [1000, 1000, 0]])
    drone = np.array([400.0, 300.0, 100.0])

    def ray(k, p):
        v = p - C[k]
        return v / np.linalg.norm(v)
    bad = ray(1, np.array([1500.0, 1800.0, 100.0]))               # cluster 1 points somewhere wrong
    free = [(0, ray(0, drone)), (1, bad), (2, ray(2, drone)), (3, ray(3, drone))]
    Track._next = 0
    trk = Tracker(C, flat=True, birth_rays=3, confirm_deg=4.0, birth_range=5000.0)
    trk.birth(0.0, free)
    assert len(trk.tracks) == 1 and np.linalg.norm(trk.tracks[0].s[:2] - drone[:2]) < 30
