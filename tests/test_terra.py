"""Terrain tool: diffraction model, screening geometry, flight paths, and the flat-ground defaults."""
import numpy as np
import pytest

from sim import acoustics as ac
from terra.dem import DEM_DIR, knife_edge_db
from terra.scenario import FlatGround, flight_path


def test_knife_edge_reference_values():
    # ITU-R P.526: 0 dB below v = -0.78, 6 dB at grazing (v = 0), ~13.9 dB at v = 1, capped at 25 dB
    j = knife_edge_db([-2.0, -0.78, 0.0, 1.0, 50.0])
    assert j[0] == 0 and j[1] == 0
    assert abs(j[2] - 6.0) < 0.1
    assert abs(j[3] - 13.9) < 0.1
    assert j[4] == 25.0


class Ridge(FlatGround):
    """A 50 m wall across x = 0 (for screening geometry without DEM files)."""

    res = 10.0

    def height(self, x, y):
        return np.where(np.abs(np.asarray(x)) < 20, 50.0, 0.0) if np.ndim(x) else (50.0 if abs(x) < 20 else 0.0)


def test_screening_blocks_behind_a_ridge_and_clears_above():
    from terra.dem import Terrain
    ridge = Ridge(1000)
    rcv = np.array([-300.0, 0.0, 0.03])
    behind = np.array([[300.0, 0.0, 10.0]])
    above = np.array([[300.0, 0.0, 400.0]])
    g_b, blk_b = Terrain.screening(ridge, behind, rcv)
    g_a, blk_a = Terrain.screening(ridge, above, rcv)
    assert blk_b[0] and not blk_a[0]
    loss_low = knife_edge_db(g_b * np.sqrt(100 / 343))[0]
    loss_high = knife_edge_db(g_b * np.sqrt(2000 / 343))[0]
    assert 10 < loss_low < loss_high <= 25                              # low tones bend round the ridge better
    assert knife_edge_db(g_a * np.sqrt(1000 / 343))[0] == 0


def test_flight_follows_terrain_and_fixed_altitude_clears_it():
    ridge = Ridge(1000)
    t, p, dur = flight_path(ridge, [[-500, 0], [500, 0]], agl=30, speed=10, alt_mode="agl")
    assert abs(dur - 105) < 1e-6
    over = (np.abs(p[:, 0]) < 5) & (t >= 0)
    assert np.all(p[over, 2] >= 52)                                     # at least 2 m over the wall
    t, p, _ = flight_path(ridge, [[-500, 0], [500, 0]], agl=30, speed=10, alt_mode="fixed")
    assert np.all(p[over, 2] >= 55) and abs(p[0, 2] - 30) < 1e-6        # fixed 30 m, raised over the wall


def test_propagation_defaults_unchanged_and_image_matches_flat_mirror():
    rng = np.random.default_rng(0)
    fs, n = 8000, 8000
    src = rng.standard_normal(n + 2 * fs).astype(np.float32)
    t = np.array([-2.0, 0.0, 1.0, 2.0])
    p = np.array([[100, 0, 50], [100, 0, 50], [110, 0, 50], [120, 0, 50]], float)
    mic = np.array([0.0, 0.0, 1.5])
    a, _ = ac.propagate_point_source(src, -2.0, fs, t, p, mic, n, 0.8, 20, 50)
    b, _ = ac.propagate_point_source(src, -2.0, fs, t, p, mic, n, 0.8, 20, 50, image=mic * [1, 1, -1])
    c, _ = ac.propagate_point_source(src, -2.0, fs, t, p, mic, n, 0.8, 20, 50,
                                     extra_loss_db=lambda f, tt: np.zeros((len(f), len(tt))))
    assert np.array_equal(a, b) and np.array_equal(a, c)
    d, _ = ac.propagate_point_source(src, -2.0, fs, t, p, mic, n, 0.8, 20, 50,
                                     extra_loss_db=lambda f, tt: np.full((len(f), len(tt)), 20.0))
    assert abs(np.std(d[2000:]) / np.std(a[2000:]) - 0.1) < 0.01         # 20 dB = x0.1


@pytest.mark.skipif(not (DEM_DIR / "glo30").exists(), reason="UAE DEM tiles not downloaded")
def test_jebel_hafeet_heights():
    from terra.dem import Terrain
    T = Terrain(24.06, 55.78, half_m=4000, res_m=20)
    assert 1000 < T.H.max() < 1300 and 200 < T.H.min() < 300             # ridge ~1,100 m over a ~250 m plain
    z, n = T.ground_plane(0, 0)
    assert n[2] < 0.95                                                   # the mountainside is steep
