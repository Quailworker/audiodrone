"""Direction finding on synthetic plane waves (tasks/phase1_localisation.md step 2)."""
import numpy as np
import pytest

from loc.doa import DOA, angle_deg
from sim.acoustics import fractional_shift
from sim.layouts import cluster_offsets

FS = 8000
DIRS = [(31.3, 12.7), (101.1, 44.6), (203.7, 5.4), (298.2, 69.9), (251.6, 24.1)]   # off the search grids


def plane_wave(mics, az, el, kind, snr_db=30, seconds=4.0, seed=0, fs=FS):
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t = np.arange(n) / fs
    if kind == "noise":
        s = rng.standard_normal(n)
    else:                                   # harmonic comb, 120 Hz fundamental, 12 harmonics
        s = sum(np.sin(2 * np.pi * 120 * k * t + rng.uniform(0, 2 * np.pi)) / k ** 0.5 for k in range(1, 13))
    a, e = np.radians(az), np.radians(el)
    u = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    x = np.stack([fractional_shift(s.astype(np.float32), -(m @ u) / 343.0, fs) for m in mics])
    x = x + rng.standard_normal(x.shape) * s.std() * 10 ** (-snr_db / 20)
    return x.astype(np.float32), u


@pytest.mark.parametrize("kind", ["tetra", "flat"])
@pytest.mark.parametrize("sig", ["noise", "comb"])
@pytest.mark.parametrize("method", ["srp", "ls"])
def test_plane_wave(kind, sig, method):
    """Within 1 deg at high SNR. Flat clusters only above 10 deg: near the horizon their elevation error
    grows as 1/sin(el) (design 7.8), which test_flat_near_horizon checks separately."""
    mics = cluster_offsets(kind)
    doa = DOA(mics, FS)
    for az, el in DIRS:
        if kind == "flat" and el < 10:
            continue
        x, u = plane_wave(mics, az, el, sig)
        res = doa.run(x, np.array([1.5, 2.0, 2.5]), methods=(method,))[method]
        err = angle_deg(res["u"], u[None])
        assert np.all(err < 1.0), (kind, sig, method, az, el, err)


def test_flat_near_horizon():
    """At 5 deg a flat cluster still gets the azimuth right; elevation is the weak part (design 7.8)."""
    mics = cluster_offsets("flat")
    doa = DOA(mics, FS)
    x, u = plane_wave(mics, 203.7, 5.4, "noise")
    res = doa.run(x, np.array([1.5, 2.0, 2.5]), methods=("srp",))["srp"]
    az_err = np.abs((res["az"] - 203.7 + 180) % 360 - 180)
    assert np.all(az_err < 1.0)
    assert np.all(angle_deg(res["u"], u[None]) < 3.0)


def test_tetra_below_horizon():
    """A tetrahedron tells above from below (design 7.3); its grid reaches -15 deg."""
    mics = cluster_offsets("tetra")
    doa = DOA(mics, FS)
    x, u = plane_wave(mics, 61.4, -9.6, "noise")
    res = doa.run(x, np.array([2.0]), methods=("srp",))["srp"]
    assert angle_deg(res["u"], u[None])[0] < 1.0


def test_4khz_matches_8khz():
    """Direction finding uses < 860 Hz, so 4 kHz audio must give the same answer (mode A option)."""
    mics = cluster_offsets("tetra")
    for fs in (8000, 4000):
        x, u = plane_wave(mics, 101.1, 44.6, "comb", fs=fs)
        res = DOA(mics, fs).run(x, np.array([1.5, 2.0]), methods=("srp",))["srp"]
        assert np.all(angle_deg(res["u"], u[None]) < 1.0), fs
