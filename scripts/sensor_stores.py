"""The microphone-cluster data as separate stores, connected by Zetaris (Border watch, design 13.8).

    python scripts/sensor_stores.py load        # fill the sector databases and the fusion database
    python scripts/sensor_stores.py register    # register each one in Zetaris as its own datasource
    python scripts/sensor_stores.py check       # count rows through Zetaris

In deployment each sector server keeps what its clusters heard, and the fusion server keeps the tracks and
alerts. The live loop (cluster -> tracker -> alerts) streams directly, because a Zetaris query takes about a
second. The stores are what Zetaris connects, for questions that span sectors and sources
(terra/federated.py).

Stores (one Postgres container "sensor-db" on the Zetaris Docker network, no published ports):
  sector_<sector>   one database per sector with incidents (e.g. sector_omn17), registered as sensors_<sector>
      clusters      cluster_id, sector, row_no, lat, lon, ground_m
      detections    cluster_id, ts, p_drone (max over the second), heard (p >= 0.5), tone_hz (median)
      bearings      cluster_id, ts, az_deg (from north), el_deg, coh   -- only seconds the bearing was used
  fusion            registered as fusion
      incidents     incident_id, name, sector, start_ts, end_ts
      tracks        incident_id, track_id, ts, lat, lon, height_agl_m, speed_m_s, heading_deg, sd_m
      alerts        incident_id, ts, tier, recipient, text

Rows are per second, from the simulated incident runs (runs_border/<id>/result.json, made by
scripts/border_incidents.py), on the watch clock the screen shows: 2026-10-25 06:00:00 GST + t0 + t.
Secrets from ~/.config/audiodrone/alerting.env (SENSOR_DB_PASSWORD, ZETARIS_*); never printed.
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from alerting.border import BorderMonitor
from alerting.config import _load_env_file
from alerting.sitedata_zetaris import connect, query
from terra.border import load as load_border
from terra.dem import metres_per_degree

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs_border"
INCIDENTS = ROOT / "terra" / "data" / "incidents"
CONTAINER = "sensor-db"
WATCH_START = dt.datetime(2026, 10, 25, 6, 0, 0)          # GST; the screen's clock starts here
SECTOR_SCHEMA = """
DROP TABLE IF EXISTS clusters, detections, bearings;
CREATE TABLE clusters   (cluster_id text PRIMARY KEY, sector text, row_no int, lat double precision,
                         lon double precision, ground_m double precision);
CREATE TABLE detections (cluster_id text, ts timestamp, p_drone real, heard boolean, tone_hz real);
CREATE TABLE bearings   (cluster_id text, ts timestamp, az_deg real, el_deg real, coh real);
"""
FUSION_SCHEMA = """
DROP TABLE IF EXISTS incidents, tracks, alerts;
CREATE TABLE incidents (incident_id text PRIMARY KEY, name text, sector text, start_ts timestamp, end_ts timestamp);
CREATE TABLE tracks    (incident_id text, track_id int, ts timestamp, lat double precision, lon double precision,
                        height_agl_m real, speed_m_s real, heading_deg real, sd_m real);
CREATE TABLE alerts    (incident_id text, ts timestamp, tier text, recipient text, text text);
"""


def sector_db(sector):
    return "sector_" + sector.lower().replace("-", "")


def source_name(sector):
    return "sensors_" + sector.lower().replace("-", "")


def ts(inc, t):
    return (WATCH_START + dt.timedelta(seconds=float(inc["t0"]) + float(t))).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def lit(v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    return repr(float(v)) if isinstance(v, float) else str(v)


def inserts(table, rows):
    return [f"INSERT INTO {table} VALUES {', '.join('(' + ', '.join(lit(v) for v in r) + ')' for r in rows[i:i + 500])};"
            for i in range(0, len(rows), 500)]


def incident_rows(inc, res, border):
    """(clusters, detections, bearings, tracks, alerts) rows of one incident, one row per second."""
    m_lat, m_lon = metres_per_degree(inc["lat"])
    def ll(x, y):
        return inc["lat"] + y / m_lat, inc["lon"] + x / m_lon
    border_xy = BorderMonitor(inc["border"], send=None)                     # signed distance to the border line
    clusters, det, bear = [], [], []
    p = np.asarray(res["p_cnn"], float)                                       # (K, F) at 10 Hz
    used = np.asarray(res["used"], bool)
    u = np.asarray(res["u"], float)                                           # (K, F, 3)
    coh = np.asarray(res["coh"], float)
    tone = np.asarray(res.get("pitch_hz", np.zeros_like(p)), float)
    t = np.asarray(res["t"])
    sec = np.floor(t).astype(int)
    for k, c in enumerate(res["clusters"]):
        idx = inc["cluster_index"][k]
        cid = f"C{idx:04d}"
        row_no = 1 if border_xy.signed_distance(c["x"], c["y"])[0] < 600 else 2   # rows 300 m and 900 m in
        lat, lon = ll(c["x"], c["y"])
        clusters.append((cid, border["clusters"][idx][2], row_no, lat, lon, c["z"]))
        for s in np.unique(sec):
            m = sec == s
            heard = bool((p[k, m] >= 0.5).any())
            tones = tone[k, m][tone[k, m] > 0]
            det.append((cid, ts(inc, s), float(p[k, m].max()), heard, float(np.median(tones)) if len(tones) else None))
            mu = m & used[k]
            if mu.any():
                v = u[k, mu].mean(0)
                v /= np.linalg.norm(v) or 1.0
                bear.append((cid, ts(inc, s), float(np.degrees(np.arctan2(v[0], v[1])) % 360),
                             float(np.degrees(np.arcsin(np.clip(v[2], -1, 1)))), float(coh[k, mu].mean())))
    from terra.dem import Terrain
    ground = Terrain(inc["lat"], inc["lon"], half_m=inc["half_m"] + 500, res_m=30.0).height
    tracks, last = [], {}
    for f in res["frames"]:
        for tr in f["tracks"]:
            s = int(f["t"])
            if last.get(tr["id"]) == s:
                continue
            last[tr["id"]] = s
            lat, lon = ll(tr["x"], tr["y"])
            tracks.append((inc["id"], tr["id"], ts(inc, f["t"]), lat, lon, tr["z"] - ground(tr["x"], tr["y"]),
                           float(np.hypot(tr["vx"], tr["vy"])), float(np.degrees(np.arctan2(tr["vx"], tr["vy"])) % 360),
                           float(max(tr["sx"], tr["sy"]))))
    alerts = [(inc["id"], ts(inc, a["t"]), a["tier"], a["to"], a["text"]) for a in res["alerts"]]
    return clusters, det, bear, tracks, alerts


def psql(db, sql):
    subprocess.run(["sg", "docker", "-c", f"docker exec -i {CONTAINER} psql -q -v ON_ERROR_STOP=1 -U sensors -d {db}"],
                   input=sql, text=True, check=True, capture_output=True)


def load():
    border = load_border()
    incs = [json.loads(f.read_text()) for f in sorted(INCIDENTS.glob("*.json"))]
    sectors = sorted({i["sector"] for i in incs})
    list_dbs = f'docker exec {CONTAINER} psql -U sensors -d fusion -Atc "SELECT datname FROM pg_database"'
    have = subprocess.run(["sg", "docker", "-c", list_dbs], capture_output=True, text=True, check=True).stdout.split()
    for s in sectors:
        if sector_db(s) not in have:
            psql("fusion", f"CREATE DATABASE {sector_db(s)};")
        psql(sector_db(s), SECTOR_SCHEMA)
    psql("fusion", FUSION_SCHEMA)
    totals = {"clusters": 0, "detections": 0, "bearings": 0, "tracks": 0, "alerts": 0}
    for inc in incs:
        res = json.loads((RUNS / inc["id"] / "result.json").read_text())
        c, d, b, tr, al = incident_rows(inc, res, border)
        db = sector_db(inc["sector"])
        indexes = "CREATE INDEX IF NOT EXISTS det_ts ON detections (ts); CREATE INDEX IF NOT EXISTS bear_ts ON bearings (ts);"
        psql(db, "\n".join(inserts("clusters", c) + inserts("detections", d) + inserts("bearings", b) + [indexes]))
        psql("fusion", "\n".join(inserts("incidents", [(inc["id"], inc["name"], inc["sector"], ts(inc, 0),
                                                        ts(inc, res["duration"]))]) + inserts("tracks", tr) + inserts("alerts", al)))
        for k, v in zip(totals, (c, d, b, tr, al), strict=True):
            totals[k] += len(v)
        print(f"{inc['id']:8s} -> {db}: {len(c)} clusters, {len(d)} detection rows, {len(b)} bearing rows; "
              f"fusion: {len(tr)} track rows, {len(al)} alerts")
    print("total", totals)


def register():
    pw = os.environ["SENSOR_DB_PASSWORD"]
    incs = [json.loads(f.read_text()) for f in sorted(INCIDENTS.glob("*.json"))]
    sources = [(source_name(s), sector_db(s), f"sector-{s}-sensors") for s in sorted({i["sector"] for i in incs})]
    sources.append(("fusion", "fusion", "fusion-server-tracks-alerts"))
    with connect() as c:
        cur = c.cursor()
        for name, db, desc in sources:
            try:
                cur.execute(f'CREATE DATASOURCE {name} DESCRIBE BY "{desc}" OPTIONS (jdbcdriver "org.postgresql.Driver", '
                            f'jdbcurl "jdbc:postgresql://{CONTAINER}:5432/{db}", username "sensors", password "{pw}")')
            except Exception as e:  # noqa: BLE001 - already registered on a rerun
                if "exist" not in str(e).lower():
                    raise RuntimeError(f"CREATE DATASOURCE {name} failed: {str(e).replace(pw, '***')[:300]}") from None
            cur.execute(f"REGISTER DATASOURCE TABLES FROM {name}")
            print(f"registered {name} ({db})")


def check():
    incs = [json.loads(f.read_text()) for f in sorted(INCIDENTS.glob("*.json"))]
    for s in sorted({i["sector"] for i in incs}):
        n = query(f"SELECT count(*) FROM {source_name(s)}.detections")[0][0]
        print(f"{source_name(s)}.detections: {n} rows through Zetaris")
    for t in ("incidents", "tracks", "alerts"):
        print(f"fusion.{t}: {query(f'SELECT count(*) FROM fusion.{t}')[0][0]} rows through Zetaris")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["load", "register", "check"])
    a = ap.parse_args()
    _load_env_file()
    {"load": load, "register": register, "check": check}[a.cmd]()


if __name__ == "__main__":
    main()
