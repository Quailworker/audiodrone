"""Site data read through Zetaris: the same tools as alerting/sitedata.py, over federated tables.

The border-ops database (posts, shelters, units; scripts/border_ops.py) is registered in Zetaris as a
JDBC datasource (CREATE DATASOURCE, REGISTER DATASOURCE TABLES), and every read goes through Zetaris
Lightning SQL over its JDBC driver as `borderops.<table>`, never to the database directly. Positions are stored as latitude/longitude for the whole border; each instance works
in the local metre frame around `origin` (an incident's origin), like the tracks.

Connection (from the environment or ~/.config/audiodrone/alerting.env):
  ZETARIS_JDBC_JAR    the driver jar (default ~/zetaris/zetaris-platform/jdbc/ndp-jdbc-driver-*.jar)
  ZETARIS_URL         default jdbc:zetaris:lightning@localhost:10000
  ZETARIS_USER, ZETARIS_PASSWORD   the account you sign in to the Zetaris web interface with
"""
import contextlib
import glob
import math
import os

from terra.dem import metres_per_degree

SOURCE = "borderops"                      # the Zetaris datasource name; tables are <source>.<table>
SEARCH_DEG = 0.12                         # bounding box queried around the origin (~13 km)


def _jar():
    if os.environ.get("ZETARIS_JDBC_JAR"):
        return os.environ["ZETARIS_JDBC_JAR"]
    hits = sorted(glob.glob(os.path.expanduser("~/zetaris/zetaris-platform/jdbc/ndp-jdbc-driver-*.jar")))
    if not hits:
        raise FileNotFoundError("Zetaris JDBC driver not found: set ZETARIS_JDBC_JAR")
    return hits[-1]


@contextlib.contextmanager
def connect():
    import jaydebeapi
    user, pw = os.environ.get("ZETARIS_USER"), os.environ.get("ZETARIS_PASSWORD")
    if not user or not pw:
        raise RuntimeError("set ZETARIS_USER and ZETARIS_PASSWORD (in ~/.config/audiodrone/alerting.env)")
    c = jaydebeapi.connect("com.zetaris.lightning.jdbc.LightningDriver",
                           os.environ.get("ZETARIS_URL", "jdbc:zetaris:lightning@localhost:10000"), [user, pw], _jar())
    try:
        yield c
    finally:
        c.close()


def query(sql):
    """One Lightning SQL statement (Zetaris runs one per call) -> list of row tuples."""
    return query_cols(sql)[1]


def query_cols(sql):
    """One Lightning SQL statement -> (column names, list of row tuples)."""
    with connect() as c:
        cur = c.cursor()
        cur.execute(sql)
        return [d[0] for d in cur.description], cur.fetchall()


class ZetarisSiteData:
    """Same tools as LocalSiteData, read from the federated border-ops tables around one origin."""

    def __init__(self, origin_latlon, fetch=query):
        self.origin = tuple(float(v) for v in origin_latlon)
        self.m_lat, self.m_lon = metres_per_degree(self.origin[0])
        self.fetch = fetch
        self._cache = {}

    def _xy(self, lat, lon):
        return round((lon - self.origin[1]) * self.m_lon, 1), round((lat - self.origin[0]) * self.m_lat, 1)

    def _rows(self, table, cols):
        if table not in self._cache:
            lat0, lon0 = self.origin
            sql = (f"SELECT {', '.join(cols)} FROM {SOURCE}.{table} WHERE lat BETWEEN {lat0 - SEARCH_DEG} AND "
                   f"{lat0 + SEARCH_DEG} AND lon BETWEEN {lon0 - SEARCH_DEG} AND {lon0 + SEARCH_DEG}")
            out = []
            for r in self.fetch(sql):
                d = dict(zip(cols, r, strict=True))
                d["x"], d["y"] = self._xy(float(d.pop("lat")), float(d.pop("lon")))
                out.append(d)
            self._cache[table] = out
        return self._cache[table]

    # --- tools ---------------------------------------------------------------------------------
    def list_sites(self):
        return self._rows("posts", ["id", "name", "sector", "lat", "lon", "radius_m"])

    def list_shelters(self):
        return self._rows("shelters", ["id", "name", "sector", "lat", "lon", "capacity"])

    def _units(self):
        return self._rows("units", ["id", "name", "kind", "sector", "lat", "lon", "size", "channel"])

    def list_groups(self):
        return self._units()

    def people_in_zone(self, x, y, radius):
        out = [{**g, "distance_m": round(math.hypot(g["x"] - x, g["y"] - y), 1)} for g in self._units()]
        return sorted((g for g in out if g["distance_m"] <= radius), key=lambda g: g["distance_m"])

    def nearest_shelter(self, x, y, avoid_x=None, avoid_y=None, avoid_radius=0.0):
        best = None
        for s in self.list_shelters():
            if avoid_x is not None and math.hypot(s["x"] - avoid_x, s["y"] - avoid_y) <= avoid_radius:
                continue
            dist = math.hypot(s["x"] - x, s["y"] - y)
            if best is None or dist < best["distance_m"]:
                best = {**s, "distance_m": round(dist, 1), "walk_min": round(dist / 80.0, 1)}
        return best or {"error": "no shelter outside the danger zone"}

    def get_group(self, group_id):
        for g in self._units():
            if g["id"] == group_id:
                return g
        return {"error": f"unknown group {group_id}"}

    def origin_latlon(self):
        return list(self.origin)
