"""Questions that span the separate stores, answered by one Zetaris query each (Border watch).

The stores (scripts/sensor_stores.py, scripts/border_ops.py), each a Zetaris datasource:
  sensors_<sector>   what one sector's clusters heard: clusters, detections, bearings
  fusion             tracks, alerts, incidents (the fusion server)
  borderops          posts, shelters, units (operations)

Each function returns {"title", "about", "sources", "sql", "columns", "rows"}; the screen shows the SQL and
the sources so it is plain that one query joined several databases. Distances use a local flat-earth
approximation (metres per degree at the track's latitude), good to well under 1 % over a few km.
"""
import datetime as dt
import json
import re
from pathlib import Path

from alerting.sitedata_zetaris import query_cols

INCIDENTS = Path(__file__).resolve().parent / "data" / "incidents"
DIST = ("SQRT(POW(({a}.lat - {b}.lat) * 110900, 2) + "
        "POW(({a}.lon - {b}.lon) * 111320 * COS(RADIANS({a}.lat)), 2))")


def _incident(iid):
    if not re.fullmatch(r"[A-Z0-9-]{1,16}", iid or ""):
        raise ValueError("bad incident id")
    f = INCIDENTS / f"{iid}.json"
    if not f.exists():
        raise ValueError(f"unknown incident {iid}")
    return json.loads(f.read_text())


def sensors(sector):
    return "sensors_" + sector.lower().replace("-", "")


def all_sectors():
    return sorted({json.loads(f.read_text())["sector"] for f in INCIDENTS.glob("*.json")})


def _run(title, about, sources, sql):
    cols, rows = query_cols(sql)
    def cell(v):
        if isinstance(v, float):
            return round(v, 2)
        if isinstance(v, (dt.datetime, dt.date)):
            return str(v)
        return v if isinstance(v, (int, str, bool)) or v is None else str(v)
    return {"title": title, "about": about, "sources": sources, "sql": " ".join(sql.split()),
            "columns": cols, "rows": [[cell(v) for v in r] for r in rows]}


def sector_overview():
    """Every sector store at once: how much each sector's clusters heard, against tracks and border alerts."""
    secs = all_sectors()
    heard = " UNION ALL ".join(
        f"SELECT '{s}' AS sector, COUNT(DISTINCT cluster_id) AS clusters, "
        f"SUM(CASE WHEN heard THEN 1 ELSE 0 END) AS cluster_seconds_heard, "
        f"MIN(CASE WHEN heard THEN ts END) AS first_heard FROM {sensors(s)}.detections" for s in secs)
    # tracks and crossings are counted in their own subqueries: joining both to the sector row multiplies them
    sql = f"""SELECT h.sector, h.clusters, h.cluster_seconds_heard, h.first_heard,
        COALESCE(tr.tracks, 0) AS tracks, COALESCE(cr.crossings, 0) AS crossings
      FROM ({heard}) h
      LEFT JOIN (SELECT i.sector, COUNT(*) AS tracks
                 FROM (SELECT DISTINCT incident_id, track_id FROM fusion.tracks) t
                 JOIN fusion.incidents i ON i.incident_id = t.incident_id GROUP BY i.sector) tr ON tr.sector = h.sector
      LEFT JOIN (SELECT i.sector, COUNT(*) AS crossings FROM fusion.alerts a
                 JOIN fusion.incidents i ON i.incident_id = a.incident_id
                 WHERE a.tier = 'border_cross' AND a.recipient = 'operators' GROUP BY i.sector) cr ON cr.sector = h.sector
      ORDER BY h.first_heard"""
    return _run("Sector overview", "What each sector's clusters heard, against the fusion server's tracks and "
                "border alerts. One query over every sector store.", [sensors(s) for s in secs] + ["fusion"], sql)


def first_heard(iid, limit=10):
    """Which clusters heard the drone first, their row, and the first bearing each measured."""
    inc = _incident(iid)
    src = sensors(inc["sector"])
    sql = f"""SELECT c.cluster_id, c.row_no, f.first_heard, ROUND(f.max_p, 2) AS max_p_drone,
        ROUND(b.az_deg) AS first_bearing_deg, ROUND(b.coh, 2) AS coherence
      FROM (SELECT d.cluster_id, MIN(d.ts) AS first_heard, MAX(d.p_drone) AS max_p
            FROM {src}.detections d JOIN fusion.incidents i ON i.incident_id = '{iid}'
                 AND d.ts BETWEEN i.start_ts AND i.end_ts
            WHERE d.heard GROUP BY d.cluster_id) f
      JOIN {src}.clusters c ON c.cluster_id = f.cluster_id
      LEFT JOIN (SELECT cluster_id, MIN(ts) AS bts FROM {src}.bearings GROUP BY cluster_id) fb
           ON fb.cluster_id = f.cluster_id
      LEFT JOIN {src}.bearings b ON b.cluster_id = fb.cluster_id AND b.ts = fb.bts
      ORDER BY f.first_heard LIMIT {int(limit)}"""
    return _run(f"First to hear {iid}", "The clusters that heard the drone first (row 1 is 300 m from the "
                "border, row 2 is 900 m) and the first bearing each one measured.", [src, "fusion"], sql)


def units_near_tracks(iid, radius_m=2000):
    """Units that any track of the incident came within radius_m of, with the closest approach."""
    _incident(iid)
    d = DIST.format(a="t", b="u")
    sql = f"""SELECT u.name AS unit, u.kind, u.size AS people, u.channel, MIN(t.ts) AS first_within,
        ROUND(MIN({d})) AS closest_m
      FROM fusion.tracks t JOIN borderops.units u ON ABS(t.lat - u.lat) < 0.05 AND ABS(t.lon - u.lon) < 0.05
      WHERE t.incident_id = '{iid}' AND {d} <= {float(radius_m)}
      GROUP BY u.name, u.kind, u.size, u.channel ORDER BY closest_m"""
    return _run(f"Units within {int(radius_m)} m of the {iid} tracks", "The fusion server's tracks joined with "
                "the operations database's units: who the drone came close to, and when.", ["fusion", "borderops"], sql)


def heard_vs_tracked(iid):
    """Seconds the sector heard a drone, and how many of them the fusion server had no track for."""
    inc = _incident(iid)
    src = sensors(inc["sector"])
    sql = f"""SELECT COUNT(*) AS seconds_heard,
        SUM(CASE WHEN t.ts_s IS NULL THEN 1 ELSE 0 END) AS seconds_heard_not_tracked,
        MIN(h.ts) AS first_heard, MIN(t.ts_s) AS first_tracked
      FROM (SELECT DISTINCT d.ts FROM {src}.detections d JOIN fusion.incidents i ON i.incident_id = '{iid}'
            AND d.ts BETWEEN i.start_ts AND i.end_ts WHERE d.heard) h
      LEFT JOIN (SELECT DISTINCT date_trunc('second', ts) AS ts_s FROM fusion.tracks WHERE incident_id = '{iid}') t
        ON t.ts_s = h.ts"""
    return _run(f"Heard against tracked, {iid}", "Seconds in which at least one cluster heard a drone, and how "
                "many of those the fusion server had no track for (missed by tracking, or a false detection).",
                [src, "fusion"], sql)


QUERIES = {"overview": sector_overview, "first_heard": first_heard, "units_near": units_near_tracks,
           "heard_vs_tracked": heard_vs_tracked}
