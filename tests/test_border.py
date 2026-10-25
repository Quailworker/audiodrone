"""Border monitor (alerting/border.py) and border geometry / deployment (terra/border.py)."""
import asyncio
import json as _json
from types import SimpleNamespace

import numpy as np
from starlette.requests import Request

from alerting.border import BorderMonitor
from alerting.core import TrackSnapshot
from terra import server
from terra.border import deploy, resample

LINE = [{"sector": "SAU-01", "neighbour": "SAU", "xy": [[-3000, 0], [3000, 0]]}]   # UAE on the left = north


def snap(tid, y, vy, sd=10.0, n=30):
    P = np.diag([sd ** 2, sd ** 2, 100.0, 1, 1, 1])
    return TrackSnapshot(tid, np.array([0.0, y, 50.0, 0.0, vy, 0.0]), P, n, 3, 0.9, 20.0)


def run(path, rate=None):
    sent = []

    def send(t, tier, to, text, **numbers):
        msg = {"t": t, "tier": tier, "to": to, "text": text, "numbers": numbers}
        sent.append(msg)
        return msg
    mon = BorderMonitor(LINE, send, rate)
    for i, (y, vy) in enumerate(path):
        mon.update(i * 0.1, [snap(1, y, vy)])
    return sent


def test_fragmented_track_reports_once():
    # one drone whose track breaks into three ids while crossing: one watch and one crossing, not three
    sent = []

    def send(t, tier, to, text, **numbers):
        sent.append(tier)
        return tier
    mon = BorderMonitor(LINE, send)
    ys = np.arange(-2500, 400, 1.5)
    for i, y in enumerate(ys):
        mon.update(i * 0.1, [snap(1 + i * 3 // len(ys), y + 20 * (i * 3 // len(ys)), 15.0)])
    assert sent.count("border_watch") == 1 and sent.count("border_cross") <= 1


def test_signed_distance_is_positive_inside():
    mon = BorderMonitor(LINE, lambda *a, **k: None)
    assert mon.signed_distance(100, 250)[0] > 249
    assert mon.signed_distance(100, -250)[0] < -249


def test_watch_then_one_crossing():
    path = [(y, 15.0) for y in np.arange(-2500, 400, 1.5)]
    sent = run(path)
    tiers = [m["tier"] for m in sent]
    assert tiers == ["border_watch", "border_cross"]
    assert "SAU-01" in sent[1]["text"] and sent[1]["numbers"]["inside_m"] >= 30


def test_track_first_seen_inside_is_not_a_crossing():
    sent = run([(y, 15.0) for y in np.arange(100, 900, 1.5)])
    assert sent == []


def test_parallel_flight_outside_is_watched_but_never_crosses():
    sent = run([(-1000.0, 0.0)] * 50)
    assert [m["tier"] for m in sent] == ["border_watch"]


def test_ground_sources_are_ignored():
    ground = SimpleNamespace(level="ground")
    sent = run([(y, 15.0) for y in np.arange(-2500, 400, 1.5)], rate=lambda tr: (ground, 15.0))
    assert sent == []


def test_deployment_rows_are_inside_and_spaced():
    # a straight east-west line at 24 N, oriented west -> east, so the UAE (left) is north
    line = [[54.0, 24.0], [54.2, 24.0]]
    c = deploy(line, spacing=600.0, offsets=(300.0, 900.0))
    north_m = (c[:, 1] - 24.0) * 110_770
    assert np.all(north_m > 250)
    assert set(np.round(north_m / 100) * 100) == {300.0, 900.0}
    ll, _, _ = resample(line, 600.0)
    step_m = np.diff(ll[:, 0]) * 101_770                        # m per degree of longitude at 24 N
    assert np.allclose(step_m, 600.0, rtol=2e-3)


def test_border_data_and_incident_routes():
    border = _json.loads((server.BORDER_DATA / "border.json").read_text())
    assert 1000 < border["length_km"] < 1050                     # UAE land border, Natural Earth
    assert {s["group"] for s in border["sectors"]} == {"SAU", "OMN", "MUS", "MAD"}
    assert len(border["clusters"]) == sum(s["clusters"] for s in border["sectors"])

    def call(fn, **path):
        return asyncio.run(fn(Request({"type": "http", "method": "GET", "path": "/", "query_string": b"",
                                       "headers": [], "path_params": path})))
    rows = _json.loads(call(server.api_incidents).body)
    assert [r["t0"] for r in rows] == sorted(r["t0"] for r in rows)                  # watch order
    assert call(server.api_incident, iid="../border").status_code == 404


def test_zetaris_site_data_maps_rows_to_the_local_frame():
    from alerting.sitedata_zetaris import ZetarisSiteData
    from terra.dem import metres_per_degree
    lat0, lon0 = 24.78, 56.15
    m_lat, m_lon = metres_per_degree(lat0)
    rows = {"posts": [("P-1", "Post A", "OMN-17", lat0 + 1000 / m_lat, lon0, 100.0)],
            "shelters": [("S-1", "Shelter B", "OMN-17", lat0, lon0 + 2000 / m_lon, 60)],
            "units": [("G-1", "Garrison A", "garrison", "OMN-17", lat0 + 1000 / m_lat, lon0, 8, "radio:a")]}
    sqls = []

    def fetch(sql):
        sqls.append(sql)
        return rows[sql.split("FROM borderops.")[1].split()[0]]
    z = ZetarisSiteData([lat0, lon0], fetch=fetch)
    post = z.list_sites()[0]
    assert (post["x"], post["y"]) == (0.0, 1000.0) and post["name"] == "Post A"
    assert z.people_in_zone(0, 1000, 50)[0]["name"] == "Garrison A"
    sh = z.nearest_shelter(0, 1000)
    assert sh["name"] == "Shelter B" and abs(sh["distance_m"] - 2236.1) < 1
    assert all("FROM borderops." in q and "BETWEEN" in q for q in sqls)       # federated tables, bounding box


def test_site_facts_give_each_post_its_shelter_and_units_near_tracks():
    from alerting.sitedata import LocalSiteData
    from terra.ask import site_facts
    site = LocalSiteData()
    core = SimpleNamespace(latest={3: {"snapshot": SimpleNamespace(state=np.array([0.0, 0.0, 50, 0, 0, 0]), n_upd=30),
                                       "rating": SimpleNamespace(level="likely")}})
    f = site_facts(site, core, unit_radius=10_000)
    assert len(f["posts"]) == len(site.list_sites())
    assert all(p["nearest_shelter"] is None or "walk_min" in p["nearest_shelter"] for p in f["posts"])
    assert f["units_within_10000_m_of_each_track"]["track 3"]
