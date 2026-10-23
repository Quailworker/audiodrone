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
