"""Border watch: site data and operator questions for an incident, at a moment of its replay.

site_data(inc)          posts, shelters and units around the incident, read through Zetaris
                        (alerting/sitedata_zetaris.py); falls back to the incident's own post when Zetaris
                        is not reachable, and says so.
ask(inc, t, question)   Nemotron's answer about the situation at incident time t (alerting/agent.py), with
                        the site tools served over MCP from Zetaris. The configured model (Nemotron 3.5 Lightning
                        through Meterless Nightshift) when it answers a short probe within BUSY_S, else Nemotron
                        Mini on this machine (Ollama).

The replay files keep each frame's tracks (position, velocity, per-axis sd) and which clusters' CNN fired,
not the full Kalman covariance or the CNN probabilities. So the alert core is rebuilt by feeding it the
frames up to t with a diagonal covariance, p_cnn 0.9 while any cluster hears the drone, and the number of
clusters hearing it: close to, but not exactly, the state the live system had.
"""
import sys
from functools import lru_cache

import numpy as np

from alerting.config import Config, _load_env_file
from alerting.core import AlertCore, TrackSnapshot
from alerting.delivery import ApprovalQueue, Delivery
from alerting.sitedata import LocalSiteData

P_HEARD, P_QUIET = 0.9, 0.3
LOCAL_LLM = {"llm_base_url": "http://127.0.0.1:11434/v1", "llm_api_key": "ollama", "llm_model": "nemotron-mini"}
PROBE_S = 8.0
BUSY_S = 3.0                        # a short reply slower than this: the model is busy, a full answer would miss its deadline


def route(cfg):
    """Where a model call goes, for the screen: Meterless Nightshift (tunnel 18088), a self-hosted server, or here."""
    url = cfg.llm_base_url
    return "meterless-nightshift" if ":18088" in url else "local-ollama" if ":11434" in url else "self-hosted"


def _responds(cfg):
    """True if the configured model answers a short request within BUSY_S. A few tokens, not one: a reasoning
    model can spend a 1-token budget thinking and come back empty."""
    import time

    from openai import OpenAI
    t0 = time.time()
    try:
        OpenAI(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key, timeout=PROBE_S, max_retries=0).chat.completions.create(
            model=cfg.llm_model, messages=[{"role": "user", "content": "Reply: ok"}], max_tokens=64,
            extra_body=cfg.llm_extra_body or None)
    except Exception:  # noqa: BLE001 - any failure means "not answering now"
        return False
    return time.time() - t0 <= BUSY_S


class _IncidentSites(LocalSiteData):
    def __init__(self, inc):
        self.d = {"sites": inc["sites"], "shelters": [], "groups": [], "origin_latlon": [inc["lat"], inc["lon"]]}


def site_backend(inc):
    """(site data object, source label). Zetaris first; the incident's own post if it is unreachable."""
    _load_env_file()
    try:
        from alerting.sitedata_zetaris import ZetarisSiteData
        z = ZetarisSiteData([inc["lat"], inc["lon"]])
        z.list_sites()                                          # one query: fails here if Zetaris is down
        return z, "zetaris"
    except Exception as e:  # noqa: BLE001 - any failure means "not available"; the screen says which
        return _IncidentSites(inc), f"local (Zetaris unavailable: {type(e).__name__})"


def site_data(inc):
    site, source = site_backend(inc)
    groups = site.list_groups() if hasattr(site, "list_groups") else []
    return {"source": source, "sites": site.list_sites(), "shelters": site.list_shelters(), "units": groups}


@lru_cache(maxsize=8)
def _terrain(lat, lon, half_m):
    from terra.dem import Terrain
    return Terrain(lat, lon, half_m=half_m + 500.0, res_m=30.0)


def core_at(inc, t, site):
    """The alert core after the replay frames up to incident time t (heights above the ground, as live)."""
    ground = _terrain(inc["lat"], inc["lon"], inc["half_m"]).height
    core = AlertCore(site, Delivery([]), ApprovalQueue(auto_after_s=15.0))
    heard_since, history = None, []
    for f in inc["frames"]:
        if f["t"] > t:
            break
        n_heard = f["cnn"].count("1")
        heard = n_heard > 0
        heard_since = (heard_since if heard_since is not None else f["t"]) if heard else None
        history.append(heard)
        duty = float(np.mean(history[-25:]))
        detected = 0.0 if heard_since is None else f["t"] - heard_since + 0.2
        snaps = []
        for tr in f["tracks"]:
            s = np.array([tr["x"], tr["y"], tr["z"] - ground(tr["x"], tr["y"]), tr["vx"], tr["vy"], tr["vz"]], float)
            P = np.diag([tr["sx"] ** 2, tr["sy"] ** 2, tr["sz"] ** 2, 4.0, 4.0, 4.0])
            snaps.append(TrackSnapshot(tr["id"], s, P, tr["n"], n_heard, P_HEARD if heard else P_QUIET, detected, duty))
        core.tick(float(f["t"]), snaps, [])
    return core


def site_facts(site, core, unit_radius=3000.0):
    """The site data a question about this moment usually needs, looked up in advance (from Zetaris when it is
    the backend): each post with its nearest shelter, and the units within unit_radius of each track. A small
    model answers from these more reliably than by choosing tool calls itself."""
    posts = []
    for p in site.list_sites():
        sh = site.nearest_shelter(p["x"], p["y"])
        posts.append({"post": p["name"], "x_east_m": p["x"], "y_north_m": p["y"],
                      "nearest_shelter": None if "error" in sh else
                      {"name": sh["name"], "distance_m": round(sh["distance_m"]), "walk_min": sh["walk_min"],
                       "capacity": sh.get("capacity")}})
    units = {}
    order = {"likely": 0, "possible": 1}                        # the most credible tracks only (fragments add noise)
    best = sorted(core.latest.items(), key=lambda kv: (order.get(kv[1]["rating"].level, 2), -kv[1]["snapshot"].n_upd))
    for tid, L in best[:3]:
        x, y = (float(v) for v in L["snapshot"].state[:2])
        units[f"track {tid}"] = [{"unit": g["name"], "people": g.get("size"), "distance_m": round(g["distance_m"])}
                                 for g in site.people_in_zone(x, y, unit_radius)]
    return {"posts": posts, f"units_within_{int(unit_radius)}_m_of_each_track": units}


def sensor_facts(inc, t):
    """What the clusters heard up to incident time t, from the sector and fusion stores through Zetaris
    (terra/federated.py). Empty if Zetaris or the stores are not there."""
    import datetime as dt

    from terra import federated
    try:
        first = federated.first_heard(inc["id"], limit=5)
        summary = federated.heard_vs_tracked(inc["id"])
    except Exception:  # noqa: BLE001 - optional context; the answer works without it
        return {}
    start = dt.datetime(2026, 10, 25, 6, 0, 0) + dt.timedelta(seconds=float(inc["t0"]))
    cut = str(start + dt.timedelta(seconds=float(t)))
    clusters = [dict(zip(first["columns"], r, strict=True)) for r in first["rows"] if str(r[2]) <= cut]
    return {"clusters_first_to_hear": [{"cluster": c["cluster_id"], "row": c["row_no"], "time": str(c["first_heard"])[11:19],
                                        "first_bearing_deg": c["first_bearing_deg"]} for c in clusters],
            "incident_hearing_summary": dict(zip(summary["columns"], summary["rows"][0], strict=True)) if summary["rows"] else {}}


async def ask(inc, t, question):
    from alerting.agent import NemotronLayer
    site, source = site_backend(inc)
    core = core_at(inc, t, site)
    facts = {**site_facts(site, core), **sensor_facts(inc, t)}
    prompt = f"{question} (Answer in one or two full sentences, naming the post or shelter and giving each number with its unit.)"
    local = Config()
    for k, v in LOCAL_LLM.items():
        setattr(local, k, v)
    local.llm_extra_body = {}                                    # Ollama's Nemotron Mini has no thinking switch
    big = Config()
    tries = ([big] if _responds(big) else []) + [local]          # Nemotron via Meterless if quick, then Nemotron Mini here
    for cfg in tries:
        if source == "zetaris":
            cfg.site_mcp = f"{sys.executable} -m alerting.sitedata_mcp --zetaris {inc['lat']},{inc['lon']}"
        async with NemotronLayer(core, cfg) as layer:
            text, log, unverified = await layer.answer(prompt, extra_facts=facts)
        if not (text or "").startswith("[system facts"):
            break
    return {"answer": text, "tools": [{"tool": x["tool"], "args": x["args"]} for x in log if isinstance(x, dict) and "tool" in x],
            "unverified": unverified, "site_source": source, "tracks": len(core.latest),
            "model": cfg.llm_model, "route": route(cfg), "fallback": bool(text and text.startswith("[system facts"))}
