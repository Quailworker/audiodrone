"""Demo border-operations data in its own database, federated through a local Zetaris (Border watch screen).

    python scripts/border_ops.py load        # create the tables in the border-ops Postgres and fill them
    python scripts/border_ops.py register    # register that database in Zetaris as a JDBC data source
    python scripts/border_ops.py check       # query it back through Zetaris

The point of Zetaris is to query an organisation's own databases where they live. So the site data
(posts, patrol units, shelters) sits in a separate Postgres ("border-ops-db", a container on the Zetaris
Docker network, no published ports), registered in Zetaris as the datasource `borderops`, and the system
reads it only through Zetaris over JDBC
(alerting/sitedata_zetaris.py).

All of it is DEMO data generated from terra/data/border.json and the incident posts: one post, one patrol
and two shelters per sector (about 2, 1 and 4 km in from the border), plus each incident's post and its
garrison. Not real positions of anything.

Secrets come from ~/.config/audiodrone/alerting.env (BORDER_OPS_DB_PASSWORD, ZETARIS_USER,
ZETARIS_PASSWORD) and are never printed.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from alerting.config import _load_env_file
from alerting.sitedata_zetaris import SOURCE, connect, query
from terra.border import load, resample
from terra.dem import metres_per_degree

ROOT = Path(__file__).resolve().parent.parent
INCIDENTS = ROOT / "terra" / "data" / "incidents"
CONTAINER = "border-ops-db"
SCHEMA = """
DROP TABLE IF EXISTS posts, shelters, units;
CREATE TABLE posts    (id text PRIMARY KEY, name text, sector text, lat double precision, lon double precision,
                       radius_m double precision);
CREATE TABLE shelters (id text PRIMARY KEY, name text, sector text, lat double precision, lon double precision,
                       capacity int);
CREATE TABLE units    (id text PRIMARY KEY, name text, kind text, sector text, lat double precision,
                       lon double precision, size int, channel text);
"""


def inward(line, frac, metres):
    """Point at fraction `frac` along a lon/lat line, `metres` into the UAE (the line has the UAE on its left)."""
    ll, tan, _ = resample(line, 200.0)
    if len(ll) == 0:
        ll, tan = np.asarray(line[:1], float), np.array([[1.0, 0.0]])
    i = min(int(frac * len(ll)), len(ll) - 1)
    m_lat, m_lon = metres_per_degree(ll[i, 1])
    north, east = tan[i, 0], -tan[i, 1]                 # left normal of the tangent (east, north)
    return float(ll[i, 1] + north * metres / m_lat), float(ll[i, 0] + east * metres / m_lon)


def rows():
    doc = load()
    posts, shelters, units = [], [], []
    for s in doc["sectors"]:
        sid = s["id"]
        lat, lon = inward(s["line"], 0.5, 2000)
        posts.append((f"P-{sid}", f"Border post {sid}", sid, lat, lon, 150.0))
        lat, lon = inward(s["line"], 0.35, 1000)
        units.append((f"U-{sid}", f"Patrol {sid}", "patrol", sid, lat, lon, 4, f"radio:{sid.lower()}"))
        for j, frac in enumerate((0.25, 0.75), 1):
            lat, lon = inward(s["line"], frac, 4000)
            shelters.append((f"S-{sid}-{j}", f"Shelter {sid}/{j}", sid, lat, lon, 60 if j == 1 else 120))
    for f in sorted(INCIDENTS.glob("*.json")):
        inc = json.loads(f.read_text())
        m_lat, m_lon = metres_per_degree(inc["lat"])
        for p in inc["sites"]:
            lat, lon = inc["lat"] + p["y"] / m_lat, inc["lon"] + p["x"] / m_lon
            posts.append((f"P-{inc['id']}", p["name"], inc["sector"], lat, lon, float(p["radius_m"])))
            units.append((f"G-{inc['id']}", f"Garrison {p['name']}", "garrison", inc["sector"], lat, lon, 8,
                          f"radio:{inc['id'].lower()}"))
    return posts, shelters, units


def sql_literal(v):
    return "'" + v.replace("'", "''") + "'" if isinstance(v, str) else repr(v)


def load_sql():
    posts, shelters, units = rows()
    out = [SCHEMA]
    for table, rs in (("posts", posts), ("shelters", shelters), ("units", units)):
        for r in rs:
            out.append(f"INSERT INTO {table} VALUES ({', '.join(sql_literal(v) for v in r)});")
    return "\n".join(out), (len(posts), len(shelters), len(units))


def docker(*args, stdin=None):
    cmd = "docker " + " ".join(args)
    return subprocess.run(["sg", "docker", "-c", cmd], input=stdin, capture_output=True, text=True, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["load", "register", "check"])
    a = ap.parse_args()
    _load_env_file()
    import os
    if a.cmd == "load":
        sql, n = load_sql()
        docker("exec", "-i", CONTAINER, "psql", "-q", "-v", "ON_ERROR_STOP=1", "-U", "borderops", "-d", "borderops", stdin=sql)
        print(f"loaded {n[0]} posts, {n[1]} shelters, {n[2]} units into {CONTAINER}")
    elif a.cmd == "register":
        pw = os.environ["BORDER_OPS_DB_PASSWORD"]
        with connect() as c:
            cur = c.cursor()
            cur.execute(f'CREATE DATASOURCE {SOURCE} DESCRIBE BY "border-ops-demo,PostgreSQL" OPTIONS ('
                        f'jdbcdriver "org.postgresql.Driver", jdbcurl "jdbc:postgresql://{CONTAINER}:5432/borderops", '
                        f'username "borderops", password "{pw}")')
            cur.execute(f"REGISTER DATASOURCE TABLES FROM {SOURCE}")
        print(f"registered datasource {SOURCE} and its tables in Zetaris")
    else:
        for table in ("posts", "shelters", "units"):
            r = query(f"SELECT count(*) FROM {SOURCE}.{table}")
            print(f"{table}: {r[0][0]} rows through Zetaris")


if __name__ == "__main__":
    main()
