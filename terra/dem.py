"""Terrain for the UAE: Copernicus DEM tiles -> a local metre grid, heights, ground slope and line of sight.

Data: Copernicus GLO-30 (1 arc-second, ~30 m) and GLO-90 DSM tiles, public on AWS
(https://copernicus-dem-30m.s3.amazonaws.com/, scripts/get_dem_uae.sh). A DSM is the top surface, so
buildings and trees count as terrain; that is what blocks sound. Tiles over open sea do not exist and
read as 0 m.

Local frame: x east, y north (m) from (lat0, lon0), z metres above sea level (EGM2008), the same axes as
the rest of the project (sim/layouts.py), so the tracker and the alert core work unchanged. Equirectangular
about lat0; within the 10 km boxes used here the scale error is < 0.1 %.
"""
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

DEM_DIR = Path(os.environ.get("DEM_DIR", Path(__file__).resolve().parent.parent / "data" / "dem"))
UAE_BOUNDS = (51.0, 22.0, 57.0, 27.0)            # lon_min, lat_min, lon_max, lat_max (tiles covered)
C_SOUND = 343.0
MAX_SCREEN_DB = 25.0                             # ISO 9613-2: barrier attenuation capped at 25 dB


def tile_path(lat, lon, res=30):
    code = 10 if res == 30 else 30
    name = f"Copernicus_DSM_COG_{code}_N{lat:02d}_00_E{lon:03d}_00_DEM"
    return DEM_DIR / f"glo{res}" / f"{name}.tif"


def metres_per_degree(lat):
    """(m per degree latitude, m per degree longitude) at latitude lat (WGS84 series)."""
    p = math.radians(lat)
    return (111132.92 - 559.82 * math.cos(2 * p) + 1.175 * math.cos(4 * p),
            111412.84 * math.cos(p) - 93.5 * math.cos(3 * p))


def read_latlon(lon_min, lat_min, lon_max, lat_max, res=30):
    """Mosaic of the tiles covering the box: (heights (rows, cols) north-up, transform)."""
    import rasterio
    from rasterio.merge import merge
    paths = [tile_path(la, lo, res) for la in range(math.floor(lat_min), math.floor(lat_max) + 1)
             for lo in range(math.floor(lon_min), math.floor(lon_max) + 1)]
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise FileNotFoundError(f"no DEM tiles in {DEM_DIR} for this area (sea, or not downloaded)")
    srcs = [rasterio.open(p) for p in paths]
    try:
        arr, tf = merge(srcs, bounds=(lon_min, lat_min, lon_max, lat_max), nodata=0.0)
    finally:
        for s in srcs:
            s.close()
    return np.nan_to_num(arr[0].astype(np.float32)), tf


class Terrain:
    """Height grid around (lat0, lon0): x, y in [-half_m, half_m] at res_m spacing (bilinear from the DEM)."""

    def __init__(self, lat0, lon0, half_m=2000.0, res_m=10.0, res=30):
        self.lat0, self.lon0, self.half, self.res = float(lat0), float(lon0), float(half_m), float(res_m)
        self.m_lat, self.m_lon = metres_per_degree(lat0)
        n = round(2 * half_m / res_m) + 1
        self.x = np.linspace(-half_m, half_m, n)
        self.y = self.x.copy()
        pad = 3 * (30 if res == 30 else 90)
        lat_a, lon_a = self.to_latlon(-half_m - pad, -half_m - pad)
        lat_b, lon_b = self.to_latlon(half_m + pad, half_m + pad)
        arr, tf = read_latlon(lon_a, lat_a, lon_b, lat_b, res)
        X, Y = np.meshgrid(self.x, self.y)                                    # H[iy, ix]
        lat, lon = self.to_latlon(X, Y)
        col = (lon - tf.c) / tf.a - 0.5                                     # pixel centres
        row = (lat - tf.f) / tf.e - 0.5
        self.H = map_coordinates(arr, [row, col], order=1, mode="nearest").astype(np.float32)

    # --- coordinates ---------------------------------------------------------------------
    def to_latlon(self, x, y):
        return self.lat0 + np.asarray(y) / self.m_lat, self.lon0 + np.asarray(x) / self.m_lon

    def to_xy(self, lat, lon):
        return (np.asarray(lon) - self.lon0) * self.m_lon, (np.asarray(lat) - self.lat0) * self.m_lat

    # --- heights ---------------------------------------------------------------------------
    def height(self, x, y):
        """Ground height (m above sea level) at x, y (arrays or scalars), bilinear; edge values outside."""
        ix = (np.asarray(x, float) + self.half) / self.res
        iy = (np.asarray(y, float) + self.half) / self.res
        h = map_coordinates(self.H, [np.atleast_1d(iy).ravel(), np.atleast_1d(ix).ravel()], order=1, mode="nearest")
        return h.reshape(np.shape(ix)) if np.ndim(ix) else float(h[0])

    def ground_plane(self, x, y, radius=20.0):
        """Least-squares plane through the ground within `radius` of (x, y): (height at x, y, unit normal)."""
        o = np.arange(-radius, radius + 1e-9, self.res / 2)
        dx, dy = np.meshgrid(o, o)
        keep = dx ** 2 + dy ** 2 <= radius ** 2
        dx, dy = dx[keep], dy[keep]
        h = self.height(x + dx, y + dy)
        A = np.column_stack([dx, dy, np.ones_like(dx)])
        (a, b, c), *_ = np.linalg.lstsq(A, h, rcond=None)
        n = np.array([-a, -b, 1.0])
        return float(c), n / np.linalg.norm(n)

    # --- line of sight ---------------------------------------------------------------------
    def screening(self, src, rcv, exclude_m=25.0, step_m=None):
        """Single knife-edge screening between source points src (N, 3) and one receiver rcv (3,).

        For each sample point i on the straight line: h_i = ground - line height (> 0 above the line),
        d1, d2 the distances to the source and the receiver. The Fresnel parameter at frequency f is
        v = h sqrt(2 f (d1 + d2) / (c d1 d2)), so the most obstructing point is the maximum of
        g_i = h_i sqrt(2 (d1 + d2) / (d1 d2)) and v(f) = g sqrt(f / c) (ITU-R P.526 single edge).
        The ground within `exclude_m` of either end is left out: the receiver's own ground is the echo
        in the propagation model, and DEM noise there would otherwise count as an obstacle.
        Returns (g (N,), blocked (N,) bool: the line passes below the ground somewhere)."""
        src = np.atleast_2d(np.asarray(src, float))
        rcv = np.asarray(rcv, float)
        step = step_m or self.res / 2
        d = src - rcv
        L = np.linalg.norm(d, axis=1)
        n = int(np.ceil(L.max() / step)) + 1
        s = np.linspace(0, 1, n)[None, :]                                   # 0 at rcv, 1 at src
        P = rcv[None, None, :] + s[..., None] * d[:, None, :]               # (N, n, 3)
        ground = self.height(P[..., 0], P[..., 1])
        h = ground - P[..., 2]
        d2 = s * L[:, None]
        d1 = L[:, None] - d2
        valid = (d1 > exclude_m) & (d2 > exclude_m)
        g = np.where(valid, h * np.sqrt(2 * (d1 + d2) / np.maximum(d1 * d2, 1e-6)), -np.inf)
        gmax = g.max(1)
        gmax[~np.isfinite(gmax)] = -1e3
        return gmax, (np.where(valid, h, -np.inf) > 0).any(1)


def knife_edge_db(v):
    """ITU-R P.526 single knife-edge diffraction loss J(v) in dB (0 for v <= -0.78), capped at 25 dB."""
    v = np.asarray(v, float)
    j = 6.9 + 20 * np.log10(np.sqrt((v - 0.1) ** 2 + 1) + v - 0.1)
    return np.clip(np.where(v > -0.78, j, 0.0), 0.0, MAX_SCREEN_DB)


@lru_cache(maxsize=1)
def overview(step_deg=0.01):
    """UAE overview heights on a regular lat/lon grid (from GLO-90): (H rows north-up, bounds)."""
    lo0, la0, lo1, la1 = UAE_BOUNDS
    arr, tf = read_latlon(lo0, la0, lo1 - 1e-6, la1 - 1e-6, res=90)
    lons = np.arange(lo0, lo1, step_deg) + step_deg / 2
    lats = np.arange(la1, la0, -step_deg) - step_deg / 2
    LO, LA = np.meshgrid(lons, lats)
    H = map_coordinates(arr, [(LA - tf.f) / tf.e - 0.5, (LO - tf.c) / tf.a - 0.5], order=1, mode="nearest")
    return H.astype(np.float32), (lo0, la0, lo1, la1)
