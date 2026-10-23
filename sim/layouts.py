"""Microphone cluster geometries and field layouts for localisation scenes (design.md 7.3, 7.8, 8.2).

A cluster is 4 mics around a centre point; a field layout places several clusters around the origin,
where the scene trajectories are centred. Positions are metres, x east, y north, z up.

    tetra  regular tetrahedron, edge a (D-07): equilateral base triangle in the horizontal plane centred on
           the cluster centre, top mic a*sqrt(2/3) above the base centroid
    flat   the tetrahedron's footprint in the ground plane (D-30): the same base triangle plus a centre mic
           where the top mic projects; all four at the mount height
"""
import numpy as np

EDGE = 0.20                       # m, D-08
MOUNT_Z = {"ground": 0.03, "pole": 2.0}
FIELDS = {
    "square5_100": [(-50, -50), (50, -50), (50, 50), (-50, 50), (0, 0)],
    "square5_150": [(-75, -75), (75, -75), (75, 75), (-75, 75), (0, 0)],
    "line3_100": [(-100, 0), (0, 0), (100, 0)],
}


def cluster_offsets(kind, a=EDGE):
    """(4, 3) mic offsets from the cluster centre (the base centroid). Mic 0 is the reference mic."""
    rc = a / np.sqrt(3)                                   # circumradius of the base triangle
    ang = np.radians([90, 210, 330])
    base = np.column_stack([rc * np.cos(ang), rc * np.sin(ang), np.zeros(3)])
    if kind == "tetra":
        top = [0.0, 0.0, a * np.sqrt(2 / 3)]
    elif kind == "flat":
        top = [0.0, 0.0, 0.0]
    else:
        raise ValueError(kind)
    return np.vstack([base, top])


def field(layout, kind, mount, a=EDGE):
    """Cluster centres (n_clusters, 3) and all mic positions (n_clusters * 4, 3), clusters in order."""
    centres = np.array([[x, y, MOUNT_Z[mount]] for x, y in FIELDS[layout]], float)
    off = cluster_offsets(kind, a)
    mics = (centres[:, None, :] + off[None]).reshape(-1, 3)
    return centres, mics


CONFIGS = {                        # tasks/phase1_localisation.md step 1.4, in priority order
    "tetra_pole_100": ("square5_100", "tetra", "pole"),
    "flat_ground_100": ("square5_100", "flat", "ground"),
    "tetra_ground_100": ("square5_100", "tetra", "ground"),
    "tetra_pole_150": ("square5_150", "tetra", "pole"),
}
