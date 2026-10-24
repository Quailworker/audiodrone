"""Harmonic frequency tracker (loc/freqtrack.py)."""
import numpy as np

from loc.freqtrack import HOP_S, REFINE_REL, HarmonicTracker

FS = 8000


def harmonic(f0, secs, rng, snr_db=10.0, n_harm=8):
    t = np.arange(int(secs * FS)) / FS
    x = sum(0.8 ** k * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 2 * np.pi)) for k in range(1, n_harm + 1))
    noise = rng.standard_normal(len(t)) * np.std(x) * 10 ** (-snr_db / 20)
    return x + noise


def centres(secs):
    return np.arange(0.5, secs - 0.5, HOP_S)


def test_tracks_a_steady_tone():
    rng = np.random.default_rng(0)
    r = HarmonicTracker(FS).run(harmonic(173.0, 8.0, rng), centres(8.0))
    f = r["f0_fine"][:, 0]
    assert np.isfinite(f).mean() > 0.9
    assert np.nanmedian(np.abs(f / 173.0 - 1)) < 0.002


def test_fine_pitch_never_leaves_the_refine_window():
    # regression: the peak bin is picked on the whitened spectrum S, so in db it can sit on a slope with
    # ~zero curvature; the parabola offset then blew up (once -209 Hz for a 178 Hz tone)
    tr = HarmonicTracker(FS)
    f0 = 178.0
    db = np.zeros((1, len(tr.f)))
    S = np.zeros_like(db)
    for k in range(1, 6):
        j = int(np.argmin(np.abs(tr.f - k * f0)))
        S[0, j] = 20.0
        db[0, j - 1], db[0, j], db[0, j + 1] = 20.0, 10.0, -1e-6         # a - 2b + c just below 0
    fine, _ = tr.refine(db, S, np.array([f0]))
    assert not np.isfinite(fine[0]) or abs(fine[0] / f0 - 1) <= REFINE_REL
