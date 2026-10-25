"""Site data: protected sites, shelters and the people and units on site (design.md 13.4).

In deployment this lives in **Zetaris** (data federation, queried in place, with its own access
control). The system reaches it through MCP tools with the names below; `alerting/sitedata_mcp.py` is a
stand-in MCP server over a local JSON site file with the same tools, so the real Zetaris can be swapped
in by pointing SITE_MCP at it (alerting/config.py).

Coordinates are metres in the local field frame (x east, y north, origin at the central cluster),
the same frame as the tracks. `origin_latlon` converts to latitude/longitude for ATAK / CoT.
"""
import json
import math
from pathlib import Path

DEFAULT_SITE = Path(__file__).with_name("example_site.json")


class LocalSiteData:
    """The JSON-file backend behind the stand-in MCP server (and used directly in tests)."""

    def __init__(self, path=DEFAULT_SITE):
        self.d = json.loads(Path(path).read_text())

    # --- tools (same names and arguments as the MCP server) ------------------------------
    def list_sites(self):
        return self.d["sites"]

    def list_shelters(self):
        return self.d["shelters"]

    def people_in_zone(self, x, y, radius):
        """Groups and units within `radius` m of (x, y)."""
        out = []
        for g in self.d["groups"]:
            dist = math.hypot(g["x"] - x, g["y"] - y)
            if dist <= radius:
                out.append({**g, "distance_m": round(dist, 1)})
        return sorted(out, key=lambda g: g["distance_m"])

    def nearest_shelter(self, x, y, avoid_x=None, avoid_y=None, avoid_radius=0.0):
        """Closest shelter to (x, y) that is not inside the avoid circle (the predicted danger zone)."""
        best = None
        for s in self.d["shelters"]:
            if avoid_x is not None and math.hypot(s["x"] - avoid_x, s["y"] - avoid_y) <= avoid_radius:
                continue
            dist = math.hypot(s["x"] - x, s["y"] - y)
            if best is None or dist < best["distance_m"]:
                best = {**s, "distance_m": round(dist, 1), "walk_min": round(dist / 80.0, 1)}  # ~80 m/min
        return best or {"error": "no shelter outside the danger zone"}

    def list_groups(self):
        return self.d["groups"]

    def get_group(self, group_id):
        for g in self.d["groups"]:
            if g["id"] == group_id:
                return g
        return {"error": f"unknown group {group_id}"}

    def origin_latlon(self):
        return self.d.get("origin_latlon", [0.0, 0.0])


def to_latlon(origin, x, y):
    """Local metres -> (lat, lon), flat-earth approximation (fine within a few km)."""
    lat0, lon0 = origin
    return lat0 + y / 111_320.0, lon0 + x / (111_320.0 * math.cos(math.radians(lat0)))
