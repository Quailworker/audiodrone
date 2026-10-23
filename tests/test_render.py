"""Layouts, the multi-mic render path and the ground truth (tasks/phase1_localisation.md step 1)."""
import itertools

import numpy as np
import pytest

from loc.render import drone_seeds, has_drone, truth
from loc.run_doa import comb_weights
from sim.layouts import CONFIGS, cluster_offsets, field
from sim.scene import sample_and_render


def test_tetrahedron_is_regular_and_flat_is_planar():
    t = cluster_offsets("tetra")
    assert np.allclose([np.linalg.norm(t[i] - t[j]) for i, j in itertools.combinations(range(4), 2)], 0.2)
    c = t - t.mean(0)
    assert np.allclose(c.T @ c, np.eye(3) * (c.T @ c)[0, 0])             # isotropic (design 7.3)
    f = cluster_offsets("flat")
    assert np.ptp(f[:, 2]) == 0 and np.allclose(f[:3], t[:3])


def test_fields():
    for layout, kind, mount in CONFIGS.values():
        centres, mics = field(layout, kind, mount)
        assert mics.shape == (4 * len(centres), 3)
        assert np.allclose(mics.reshape(len(centres), 4, 3)[:, 0, :2] - centres[:, None, :2][:, 0], cluster_offsets(kind)[0, :2])


def test_drone_seeds_mirror_the_scene():
    seeds = drone_seeds(5)
    assert all(has_drone(s) for s in seeds) and seeds == sorted(seeds) and seeds[0] >= 20_000_000


@pytest.mark.slow
def test_label_mics_path_matches_default():
    """With one explicit mic, the multi-mic path must give the same audio and labels as the default path
    (same random draws; only the float summation order of the categories differs)."""
    seed = drone_seeds(1)[0]
    mic = [(3.0, -4.0, 1.5)]
    a0, l0, m0 = sample_and_render(seed, False, mic)
    a1, l1, m1, tracks = sample_and_render(seed, False, mic, label_mics=[0])
    assert m0["duration"] == m1["duration"] and len(tracks) == len([s for s in m0["sources"] if s["category"].startswith("drone")])
    assert np.max(np.abs(a0 - a1)) <= 1e-6 * np.max(np.abs(a0))
    for k in l0:
        assert np.allclose(l0[k].astype(float), l1[0][k].astype(float), atol=0.05), k


def test_truth_geometry_static_source():
    """A hovering drone: the true direction points at it, range is right and radial speed is zero."""
    t = np.arange(-10, 31, 0.02)
    p = np.tile([30.0, 40.0, 50.0], (len(t), 1))
    shaft = np.full((4, len(t)), 100.0)
    q = {"blades": 2}
    centres = np.array([[0.0, 0.0, 2.0], [100.0, 0.0, 2.0]])
    n = 300
    lab = {"frame_t": np.arange(3000) / 100, "snr_lines/drone_electric": np.full((3000, 4), 10.0, np.float16)}
    for cat in ("drone_electric", "bed"):
        lab[f"bands/{cat}"] = np.full((48, n * 25 // 10), -40.0 if cat == "drone_electric" else -50.0, np.float16)
    tr = truth([(q, t, p, shaft)], centres, [lab, lab], 30.0)
    for k, c in enumerate(centres):
        r = np.linalg.norm(p[0] - c)
        assert np.allclose(tr[f"d0/c{k}/u"], (p[0] - c) / r, atol=1e-5)
        assert np.allclose(tr[f"d0/c{k}/r"], r, atol=1e-3)
        assert np.allclose(tr[f"d0/c{k}/rdot"], 0, atol=1e-4)
        assert np.allclose(tr[f"d0/c{k}/tau"], tr["frame_t"] - r / 343.0, atol=1e-4)
    assert np.allclose(tr["c0/drone_electric/snr_df860"], 10.0, atol=0.01)         # -40 dB vs -50 dB


def test_comb_weights_follow_doppler():
    """Bins near multiples of the received shaft rate f0 / (1 + rdot / c) are kept, others not."""
    f = np.arange(100, 860, 2.0)
    shaft = np.full((1, 100), 120.0)
    tr = {"frame_t": np.array([1.0]), "d0/t": np.linspace(0, 10, 100), "d0/shaft_hz": shaft,
          "d0/c0/tau": np.array([0.9]), "d0/c0/rdot": np.array([-34.3])}            # approaching at c/10
    w = comb_weights(tr, 0, f, tol_hz=1.0)[0]
    f0 = 120.0 / (1 - 0.1)
    expect = np.zeros_like(f, bool)
    for n in range(1, 8):
        expect |= np.abs(f - n * f0) <= 1.0
    assert np.array_equal(w > 0, expect)
