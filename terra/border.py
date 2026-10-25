"""UAE land border: geometry, sectors and a sensor deployment along it (Border watch screen).

    python -m terra.border build --ne <ne_10m_admin_0_boundary_lines_land.geojson>   -> terra/data/border.json

Border: Natural Earth 1:10m admin-0 land boundary lines (public domain, v5.1.2), every line with the UAE
on one side: Saudi Arabia (~455 km), Oman (Al Ain to Hatta and Fujairah, ~450 km), Musandam (~75 km) and
the Madha / Nahwa enclave rings. Natural Earth marks these lines "(verify)"; they are good to a few hundred
metres, fine for a demo map, not for siting real hardware.

Each line is stored with the UAE on its left (Natural Earth gives ADM0_LEFT / ADM0_RIGHT), so the inward
normal is the left normal. Lines are cut into sectors of about SECTOR_KM along the border, named by the
neighbour (SAU, OMN, MUS for Musandam, MAD for Madha) and numbered along the border.

Deployment (D-30 flat ground clusters): ROWS rows inside the UAE, parallel to the border, OFFSETS_M in
from it, a cluster every SPACING_M along each row, the rows staggered by half a spacing so the two rows
form triangles (the spacing study, results/spacing, favours 300-600 m over 100 m).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from .dem import metres_per_degree

DATA = Path(__file__).resolve().parent / "data"
SECTOR_KM = 25.0
SPACING_M = 600.0
OFFSETS_M = (300.0, 900.0)                       # inward distance of each row from the border line


def _ll_to_xy(lon, lat, lat0, lon0):
    m_lat, m_lon = metres_per_degree(lat0)
    return (np.asarray(lon) - lon0) * m_lon, (np.asarray(lat) - lat0) * m_lat


def _length_m(line):
    a = np.asarray(line, float)
    lat0 = float(a[:, 1].mean())
    x, y = _ll_to_xy(a[:, 0], a[:, 1], lat0, float(a[:, 0].mean()))
    return float(np.hypot(np.diff(x), np.diff(y)).sum())


def uae_lines(geojson):
    """All boundary lines touching the UAE, each [[lon, lat], ...] oriented with the UAE on the left,
    with the neighbour's ISO code."""
    out = []
    for f in geojson["features"]:
        p = f["properties"]
        if "ARE" not in (p["ADM0_A3_L"], p["ADM0_A3_R"]):
            continue
        g = f["geometry"]
        parts = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
        other = p["ADM0_A3_R"] if p["ADM0_A3_L"] == "ARE" else p["ADM0_A3_L"]
        for ln in parts:
            ln = [[float(a), float(b)] for a, b in ln]
            out.append({"line": ln if p["ADM0_A3_L"] == "ARE" else ln[::-1], "other": other})
    return out


def chain(lines):
    """Join lines that share end points (same neighbour) into longer chains, keeping UAE-on-the-left:
    a chain is only extended by a line whose start is the chain's end (both keep the same orientation)."""
    def key(p):
        return (round(p[0], 5), round(p[1], 5))
    todo = [dict(x) for x in lines]
    chains = []
    while todo:
        cur = todo.pop(0)
        grown = True
        while grown:
            grown = False
            for i, o in enumerate(todo):
                if o["other"] != cur["other"]:
                    continue
                if key(o["line"][0]) == key(cur["line"][-1]):
                    cur["line"] = cur["line"] + o["line"][1:]
                elif key(o["line"][-1]) == key(cur["line"][0]):
                    cur["line"] = o["line"] + cur["line"][1:]
                else:
                    continue
                todo.pop(i)
                grown = True
                break
        chains.append(cur)
    return chains


def group_of(ch):
    if ch["other"] == "SAU":
        return "SAU"
    a = np.asarray(ch["line"])
    if a[:, 1].mean() > 25.55:
        return "MUS"
    closed = np.allclose(a[0], a[-1], atol=1e-4)
    if closed or (56.2 < a[:, 0].mean() < 56.32 and 25.19 < a[:, 1].mean() < 25.34):
        return "MAD"
    return "OMN"


def resample(line, step_m):
    """Points every step_m along a lon/lat polyline, with the unit tangent (east, north) at each."""
    a = np.asarray(line, float)
    lat0, lon0 = float(a[:, 1].mean()), float(a[:, 0].mean())
    m_lat, m_lon = metres_per_degree(lat0)
    x, y = _ll_to_xy(a[:, 0], a[:, 1], lat0, lon0)
    seg = np.hypot(np.diff(x), np.diff(y))
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] < step_m:
        return np.empty((0, 2)), np.empty((0, 2)), s[-1]
    q = np.arange(step_m / 2, s[-1], step_m)
    qx, qy = np.interp(q, s, x), np.interp(q, s, y)
    j = np.clip(np.searchsorted(s, q) - 1, 0, len(seg) - 1)
    tx, ty = np.diff(x)[j] / np.maximum(seg[j], 1e-9), np.diff(y)[j] / np.maximum(seg[j], 1e-9)
    ll = np.column_stack([lon0 + qx / m_lon, lat0 + qy / m_lat])
    return ll, np.column_stack([tx, ty]), s[-1]


def deploy(line, spacing=SPACING_M, offsets=OFFSETS_M):
    """Cluster lon/lat for one UAE-on-the-left line: rows offset inward along the left normal."""
    out = []
    for r, off in enumerate(offsets):
        ll, tan = _shift(line, spacing) if r % 2 else resample(line, spacing)[:2]   # alternate rows staggered
        if len(ll) == 0:
            continue
        lat = ll[:, 1]
        m_lat = np.array([metres_per_degree(v)[0] for v in lat])
        m_lon = np.array([metres_per_degree(v)[1] for v in lat])
        nx, ny = -tan[:, 1], tan[:, 0]              # left normal = into the UAE
        out.append(np.column_stack([ll[:, 0] + nx * off / m_lon, ll[:, 1] + ny * off / m_lat]))
    return np.vstack(out) if out else np.empty((0, 2))


def _shift(line, spacing):
    """Like resample(line, spacing) but starting a half spacing later (the staggered row)."""
    ll, tan, _ = resample(line, spacing / 2)
    return ll[1::2], tan[1::2]


def sectors(chains):
    """Cut each chain into sectors of about SECTOR_KM; number each group along the border."""
    out = []
    counters = {}
    order = sorted(chains, key=lambda c: (group_of(c), -np.mean(np.asarray(c["line"])[:, 1]) if group_of(c) != "SAU"
                                          else np.mean(np.asarray(c["line"])[:, 0])))
    for ch in order:
        g = group_of(ch)
        a = np.asarray(ch["line"], float)
        L = _length_m(a)
        n = max(1, round(L / (SECTOR_KM * 1000)))
        lat0, lon0 = float(a[:, 1].mean()), float(a[:, 0].mean())
        x, y = _ll_to_xy(a[:, 0], a[:, 1], lat0, lon0)
        s = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
        cuts = np.linspace(0, s[-1], n + 1)
        for i in range(n):
            keep = (s >= cuts[i]) & (s <= cuts[i + 1])
            idx = np.flatnonzero(keep)
            lo, hi = max(idx.min() - 1, 0) if len(idx) else 0, min(idx.max() + 1, len(a) - 1) if len(idx) else len(a) - 1
            part = [[float(np.interp(cuts[i], s, a[:, 0])), float(np.interp(cuts[i], s, a[:, 1]))]] + \
                   a[lo + 1:hi].tolist() + \
                   [[float(np.interp(cuts[i + 1], s, a[:, 0])), float(np.interp(cuts[i + 1], s, a[:, 1]))]]
            counters[g] = counters.get(g, 0) + 1
            out.append({"id": f"{g}-{counters[g]:02d}", "group": g, "neighbour": ch["other"], "line": part,
                        "length_km": round(_length_m(part) / 1000, 1)})
    return out


GROUP_NAMES = {"SAU": "UAE-Saudi Arabia", "OMN": "UAE-Oman", "MUS": "UAE-Oman (Musandam)", "MAD": "Madha / Nahwa enclaves"}


def build(ne_path, out_path=DATA / "border.json"):
    geo = json.loads(Path(ne_path).read_text())
    secs = sectors(chain(uae_lines(geo)))
    clusters = []
    for s in secs:
        for lon, lat in deploy(s["line"]):
            clusters.append([round(float(lon), 5), round(float(lat), 5), s["id"]])
        s["clusters"] = sum(1 for c in clusters if c[2] == s["id"])
    total_km = sum(s["length_km"] for s in secs)
    doc = {"source": "Natural Earth 1:10m admin-0 land boundary lines v5.1.2 (public domain); lines marked (verify)",
           "spacing_m": SPACING_M, "row_offsets_m": list(OFFSETS_M), "length_km": round(total_km, 1),
           "groups": GROUP_NAMES,
           "sectors": [{**s, "line": [[round(a, 5), round(b, 5)] for a, b in s["line"]]} for s in secs],
           "clusters": clusters}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, separators=(",", ":")))
    return doc


def load():
    return json.loads((DATA / "border.json").read_text())


def nearest_border_point(doc, lat, lon):
    """(sector, lon, lat, unit tangent east/north with the UAE on the left) of the border point nearest (lat, lon)."""
    best = None
    for s in doc["sectors"]:
        a = np.asarray(s["line"], float)
        x, y = _ll_to_xy(a[:, 0], a[:, 1], lat, lon)
        for i in range(len(a) - 1):
            p, d = np.array([x[i], y[i]]), np.array([x[i + 1] - x[i], y[i + 1] - y[i]])
            L2 = float(d @ d)
            if L2 == 0:
                continue
            u = float(np.clip(-(p @ d) / L2, 0, 1))
            q = p + u * d
            r = float(np.hypot(*q))
            if best is None or r < best[0]:
                best = (r, s, q, d / math.sqrt(L2))
    _, s, q, t = best
    m_lat, m_lon = metres_per_degree(lat)
    return s, lon + q[0] / m_lon, lat + q[1] / m_lat, t


def local_border(doc, lat0, lon0, half_m):
    """Border lines within the box around (lat0, lon0) in local metres (UAE on the left), with sector ids."""
    out = []
    for s in doc["sectors"]:
        a = np.asarray(s["line"], float)
        x, y = _ll_to_xy(a[:, 0], a[:, 1], lat0, lon0)
        near = (np.abs(x) <= half_m * 1.5) & (np.abs(y) <= half_m * 1.5)
        if near.any():
            i0, i1 = max(np.flatnonzero(near).min() - 1, 0), min(np.flatnonzero(near).max() + 2, len(a))
            out.append({"sector": s["id"], "neighbour": s["neighbour"],
                        "xy": np.column_stack([x[i0:i1], y[i0:i1]]).round(1).tolist()})
    return out


def local_clusters(doc, lat0, lon0, radius_m, limit=24):
    """Deployment clusters within radius_m of (lat0, lon0): (local xy (N, 2), their indices in doc)."""
    c = np.asarray([[a, b] for a, b, _ in doc["clusters"]], float)
    x, y = _ll_to_xy(c[:, 0], c[:, 1], lat0, lon0)
    r = np.hypot(x, y)
    idx = np.argsort(r)[:limit]
    idx = idx[r[idx] <= radius_m]
    return np.column_stack([x[idx], y[idx]]), idx


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--ne", required=True, help="ne_10m_admin_0_boundary_lines_land.geojson")
    a = ap.parse_args()
    doc = build(a.ne)
    by = {}
    for s in doc["sectors"]:
        by.setdefault(s["group"], [0, 0.0, 0])
        by[s["group"]][0] += 1
        by[s["group"]][1] += s["length_km"]
        by[s["group"]][2] += s["clusters"]
    for g, (n, km, nc) in by.items():
        print(f"{g}: {n} sectors, {km:.0f} km, {nc} clusters")
    print(f"total {doc['length_km']} km, {len(doc['clusters'])} clusters")


if __name__ == "__main__":
    main()
