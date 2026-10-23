"""Alerting layer: rating, danger zone, alert core, delivery, site-data MCP server, Nemotron guards."""
import asyncio
import json
import sys
import urllib.request

import numpy as np
import pytest

from alerting import danger
from alerting.agent import NemotronLayer, check_numbers, check_units
from alerting.core import COVER_UPDATES, AlertCore, Detection, TrackSnapshot
from alerting.delivery import ApprovalQueue, Delivery, cot_event
from alerting.rating import Evidence, rate
from alerting.sitedata import LocalSiteData


# --- rating (D-32, D-38) -----------------------------------------------------------------

def test_rating_levels():
    assert rate(Evidence(0.9, 10, 1.0, height=1.0, sigma_h=2, speed=0.5, n_clusters=3)).level == "ground"     # chainsaw-like
    assert rate(Evidence(0.9, 10, 1.0, height=60, sigma_h=3, speed=12, n_clusters=3)).level == "likely"
    assert rate(Evidence(0.9, 10, 1.0, height=40, sigma_h=3, speed=0.3, n_clusters=3)).level == "likely"     # hovering
    assert rate(Evidence(0.9, 10, 1.0)).level == "possible"                                                  # no position
    r = rate(Evidence(0.9, 10, 1.0, height=40, sigma_h=25, speed=10))
    assert r.level == "possible" and r.likelihood is None                                                    # height too uncertain
    assert rate(Evidence(0.9, 0.2, 1.0, height=60, sigma_h=3, speed=12)).level != "likely"                   # too brief


# --- danger zone (13.3) --------------------------------------------------------------------

def _P(pos_sd=3.0, vel_sd=1.0):
    return np.diag([pos_sd ** 2] * 3 + [vel_sd ** 2] * 3)


def test_danger_head_on_arrival_time():
    th = danger.assess([300, 0, 50, -10, 0, 0], _P(), (0, 0), 60)
    assert th.threatened and 18 < th.tau_arrive < 26 and th.d_min < 1          # (300 - 60) / 10 = 24 s, less spread


def test_danger_moving_away_or_passing_wide():
    assert not danger.assess([100, 0, 50, 10, 0, 0], _P(), (0, 0), 60).threatened
    assert not danger.assess([300, 400, 50, -10, 0, 0], _P(), (0, 0), 60).threatened


def test_danger_uncertain_far_future_is_not_now():
    """Regression (replay L20000014): a slow drone 200 m away with an uncertain future must not read as
    'over the site now' (the old R + k sigma(tau*) form did)."""
    th = danger.assess([0, 200, 60, 0, -1, 0], _P(8, 3), (0, 0), 60)
    assert th.tau_arrive is None or th.tau_arrive > 60


def test_prob_inside():
    assert danger.prob_inside(0.0, 1.0, 10.0) > 0.99
    assert danger.prob_inside(100.0, 1.0, 10.0) < 1e-6


# --- alert core (13.4) ---------------------------------------------------------------------

class Collect:
    def __init__(self):
        self.msgs = []

    def send(self, m):
        self.msgs.append(m)


def _core(auto_after_s=None):
    sink = Collect()
    core = AlertCore(LocalSiteData(), Delivery([sink]), ApprovalQueue(auto_after_s))
    return core, sink


def _snap(x, y, vx, vy, n_upd=50, z=60.0, tid=1, sd=3.0):
    return TrackSnapshot(tid, np.array([x, y, z, vx, vy, 0.0]), _P(sd), n_upd, 3, 0.9, 20.0)


def test_core_take_cover_once_then_repeat_and_all_clear():
    core, sink = _core()
    core.tick(0.0, [_snap(150, 0, -12, 0)])
    tc = [m for m in sink.msgs if m["tier"] == "take_cover"]
    assert {m["to"] for m in tc} >= {"U1", "operators"}
    n = len(sink.msgs)
    core.tick(0.1, [_snap(148.8, 0, -12, 0)])
    assert len(sink.msgs) == n                                     # no repeat within REPEAT_S
    core.tick(1.0, [])                                             # track lost
    assert sink.msgs[-1]["tier"] == "all_clear"


def test_core_new_track_is_advisory_only():
    core, sink = _core()
    core.tick(0.0, [_snap(150, 0, -12, 0, n_upd=COVER_UPDATES - 1)])
    assert all(m["tier"] == "advisory" for m in sink.msgs) and sink.msgs


def test_core_uncertain_course_never_says_no_site():
    core, sink = _core()
    core.tick(0.0, [_snap(1400, 120, -10, 0, n_upd=COVER_UPDATES - 1)])
    assert sink.msgs and all("No protected site" not in m["text"] for m in sink.msgs)
    assert "Village" in sink.msgs[0]["text"]


def test_core_ground_source_is_only_logged():
    core, sink = _core()
    core.tick(0.0, [_snap(20, 0, 0.2, 0, z=1.0)])
    assert sink.msgs == [] and core.events[-1]["tier"] == "log"


def test_core_detection_without_track_one_advisory():
    core, sink = _core()
    core.tick(0.0, [], [Detection(0, 0.9, 1.0), Detection(2, 0.9, 1.0)])
    assert len(sink.msgs) == 1 and sink.msgs[0]["tier"] == "advisory"


def test_core_evacuation_needs_approval_when_time_allows():
    core, sink = _core(auto_after_s=5)
    core.tick(0.0, [_snap(1400, 120, -10, 0)])                     # village (220, 120) ~ 118 s away
    t = 0.0
    x = 1400.0
    while t < 30 and not any(m["tier"] == "approval" for m in sink.msgs):
        t += 0.1
        x -= 1.0
        core.tick(t, [_snap(x, 120, -10, 0)])
    assert any(m["tier"] == "approval" for m in sink.msgs)
    assert not any(m["tier"] == "evacuate" for m in sink.msgs)
    for _ in range(60):                                            # auto-approval after 5 s (replay setting)
        t += 0.1
        x -= 1.0
        core.tick(t, [_snap(x, 120, -10, 0)])
    ev = [m for m in sink.msgs if m["tier"] == "evacuate"]
    assert ev and all("Shelter" in m["text"] for m in ev)


def test_cot_xml():
    xml = cot_event(3, 100, 200, 50, (-37.8, 144.96))
    assert xml.startswith("<event") and 'type="a-h-A-M-F-Q"' in xml and "lat=" in xml


# --- Nemotron guards ---------------------------------------------------------------------

def test_number_guard():
    assert check_numbers("Go to Shelter B, 141 m, about 2 min.", [141.0, 2.0]) == []
    assert check_numbers("Shelter B is 100 m away.", [141.0]) == [100.0]
    assert check_numbers("Unit U1 go now.", []) == []                       # ids are not numbers
    assert check_numbers("Likelihood 90%.", [0.9]) == []
    from alerting.agent import normalise
    assert check_numbers(normalise("Track at (45 m E, \u2010443 m N)."), [45.0, -443.0]) == []


def test_unit_guard():
    assert check_units("Shelter A, 235 m, 3 min, drone in 4 min.", {"s": [4], "min": [3], "m": [235]}) == [(4.0, "min")]
    assert check_units("Drone in 2 minutes.", {"s": [120]}) == []


def test_name_guard():
    from alerting.agent import normalise
    core, _ = _core()
    layer = NemotronLayer.__new__(NemotronLayer)
    layer.core = core
    assert layer.unknown_names("Go to Shelter B now.") == []
    assert layer.unknown_names(normalise("Go to Shelter\u202fE, 150\u202fm.")) == []        # typographic spaces
    assert layer.unknown_names("Go to West Shelter now.") == ["West Shelter"]


# --- site-data MCP server (the Zetaris slot) ------------------------------------------------

def test_site_mcp_server_roundtrip():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def go():
        params = StdioServerParameters(command=sys.executable, args=["-m", "alerting.sitedata_mcp"])
        async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
            await s.initialize()
            names = {t.name for t in (await s.list_tools()).tools}
            res = await s.call_tool("nearest_shelter", {"x": 210, "y": 170, "avoid_x": 220, "avoid_y": 120, "avoid_radius": 80})
            return names, json.loads(res.content[0].text)

    names, shelter = asyncio.run(go())
    assert {"people_in_zone", "nearest_shelter", "list_shelters", "list_sites", "get_group"} <= names
    assert shelter["name"] != "Shelter C"                            # C is inside the avoided village zone


def _ollama_up():
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1)
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _ollama_up(), reason="no local OpenAI-compatible endpoint")
@pytest.mark.slow
def test_nemotron_followup_is_grounded():
    """With a local Nemotron: a follow-up is either rejected (None) or names the right shelter and only
    supported numbers. Never a made-up shelter."""
    core, _ = _core(auto_after_s=1)
    for i in range(40):
        core.tick(i * 0.1, [_snap(1400 - i, 120, -10, 0)])
    alert = next(m for m in core.events if m["tier"] == "evacuate" and m["to"] == "C1")

    async def go():
        async with NemotronLayer(core) as layer:
            return await layer.personalise(alert, core.site.get_group("C1")), layer

    out, layer = asyncio.run(go())
    if out is not None:
        assert "Shelter" in out["text"] and layer.unknown_names(out["text"]) == []
