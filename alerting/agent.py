"""Nemotron layer (design.md 13.4, D-34): messages for people, operator questions, summaries.

Nemotron runs behind an OpenAI-compatible endpoint (Meterless Nightshift on site; local Ollama as the
stand-in, alerting/config.py) and calls tools:

  core tools   get_tracks, get_danger_zone, explain_evidence     (alerting/core.py, in process)
  site tools   people_in_zone, nearest_shelter, list_shelters,   (MCP: Zetaris, or the stand-in
               list_sites, get_group                               alerting/sitedata_mcp.py)

Rules (13.4):
  - The core has already sent every alert. This layer only adds follow-ups and answers; if it is slow,
    down or wrong, nothing is lost.
  - Every number in a message must come from the alert or a tool result (`check_numbers`). Otherwise
    the message is dropped and the template stands.
  - Every prompt, tool call and answer is logged.
"""
import asyncio
import json
import re
import time
from contextlib import AsyncExitStack

import numpy as np
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

from alerting import danger
from alerting.config import Config

SYSTEM = ("You write alerts and answers for an acoustic drone-warning system protecting people. "
          "Use the tools for every fact. Never invent numbers, places or names; only use numbers that a tool "
          "returned or that are in the alert. Be short, calm and specific: what to do, where to go, how long.")
NUM = re.compile(r"(?<![A-Za-z0-9_.])-?\d+(?:\.\d+)?(?![A-Za-z0-9_])")


UNIT = re.compile(r"(?<![A-Za-z0-9_.])(-?\d+(?:\.\d+)?)\s*(seconds?|secs?|s|minutes?|mins?|metres?|meters?|m)\b", re.I)
UNIT_KIND = {"s": "s", "sec": "s", "secs": "s", "second": "s", "seconds": "s", "min": "min", "mins": "min",
             "minute": "min", "minutes": "min", "m": "m", "metre": "m", "metres": "m", "meter": "m", "meters": "m"}


def check_units(text, facts_by_unit):
    """Numbers stated with a unit must match a fact of that unit: {"s": [...], "min": [...], "m": [...]}.
    Returns the offending (value, unit) pairs."""
    bad = []
    for v, u in UNIT.findall(text or ""):
        kind = UNIT_KIND[u.lower()]
        pool = list(facts_by_unit.get(kind, []))
        if kind == "min":
            pool += [x / 60.0 for x in facts_by_unit.get("s", [])]
        if not any(abs(float(v) - a) <= max(0.51, 0.02 * abs(a)) for a in pool):
            bad.append((float(v), kind))
    return bad


def numbers_in(text):
    return [float(m) for m in NUM.findall(text or "")]


def check_numbers(text, allowed):
    """Numbers in `text` that are not (a rounding of) a number in `allowed`. Empty list = OK.
    Seconds may be stated as minutes (a/60) and probabilities as percentages (100a)."""
    pool = set()
    for a in allowed:
        pool.update({a, a / 60.0})
        if 0 <= a <= 1:
            pool.add(100 * a)
    bad = []
    for n in numbers_in(text):
        if not any(abs(n - a) <= max(0.51, 0.02 * abs(a)) for a in pool):
            bad.append(n)
    return bad


def _collect_numbers(obj, out):
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        out.extend(numbers_in(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_numbers(v, out)
    return out


class NemotronLayer:
    def __init__(self, core, cfg: Config | None = None, log=None):
        self.core = core
        self.cfg = cfg or Config()
        self.client = OpenAI(base_url=self.cfg.llm_base_url, api_key=self.cfg.llm_api_key, timeout=self.cfg.llm_timeout_s)
        self.log = log                        # a delivery sink for the audit trail (LogSink)
        self.stack = AsyncExitStack()
        self.session = None
        self.site_tools = []

    # --- lifecycle -----------------------------------------------------------------------
    async def __aenter__(self):
        cmd, args = self.cfg.site_mcp_command()
        read, write = await self.stack.enter_async_context(stdio_client(StdioServerParameters(command=cmd, args=args)))
        self.session = await self.stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        self.site_tools = (await self.session.list_tools()).tools
        return self

    async def __aexit__(self, *exc):
        await self.stack.aclose()

    # --- tools ---------------------------------------------------------------------------
    def tool_specs(self, names=None):
        local = [
            ("get_tracks", "Current drone tracks: id, position (m east/north), height (m), speed (m/s), rating.", {}),
            ("get_danger_zone", "Predicted danger for one track: for each protected site the time to arrival (s), "
             "closest pass (m) and zone radius (m), plus the predicted path.", {"track_id": {"type": "integer"}}),
            ("explain_evidence", "Why a track is rated as it is: CNN probability, height, speed, persistence, "
             "clusters hearing it.", {"track_id": {"type": "integer"}}),
        ]
        specs = [{"type": "function", "function": {"name": n, "description": d,
                  "parameters": {"type": "object", "properties": p, "required": list(p)}}} for n, d, p in local]
        specs += [{"type": "function", "function": {"name": t.name, "description": t.description or "",
                   "parameters": t.input_schema}} for t in self.site_tools]
        return [s for s in specs if names is None or s["function"]["name"] in names]

    async def call_tool(self, name, args):
        if name == "get_tracks":
            return [self._track_view(tid) for tid in self.core.latest]
        if name in ("get_danger_zone", "explain_evidence"):
            tid = int(args.get("track_id", -1))
            if tid not in self.core.latest:
                return {"error": f"no current track {tid}"}
            return self._danger_view(tid) if name == "get_danger_zone" else self._evidence_view(tid)
        res = await self.session.call_tool(name, args)
        text = "".join(getattr(c, "text", "") for c in res.content)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"result": text}

    def _track_view(self, tid):
        L = self.core.latest[tid]
        s = L["snapshot"]
        return {"track_id": tid, "x_east_m": round(float(s.state[0])), "y_north_m": round(float(s.state[1])),
                "height_m": round(float(s.state[2])), "speed_m_s": round(L["speed"], 1), "rating": L["rating"].level,
                "likelihood": None if L["rating"].likelihood is None else round(L["rating"].likelihood, 2)}

    def _danger_view(self, tid):
        L = self.core.latest[tid]
        s = L["snapshot"]
        sites = [{"site": th.site, "threatened": th.threatened, "distance_now_m": round(th.distance_now),
                  "closest_pass_m": round(th.d_min), "seconds_to_closest": round(th.tau_min),
                  "seconds_to_arrival": None if th.tau_arrive is None else round(th.tau_arrive)} for th in L["threats"]]
        path = [{"t_s": round(p[0]), "x_m": round(p[1]), "y_m": round(p[2]), "half_width_m": round(p[3])}
                for p in danger.corridor(s.state, s.P, 0.0, t_max=60, step=20)]
        return {"track_id": tid, "sites": sites, "predicted_path": path}

    def _evidence_view(self, tid):
        L = self.core.latest[tid]
        s = L["snapshot"]
        r = L["rating"]
        return {"track_id": tid, "rating": r.level, "likelihood": None if r.likelihood is None else round(r.likelihood, 2),
                "factors": {k: round(v, 2) for k, v in r.factors.items()}, "reasons": r.reasons,
                "clusters_hearing": s.n_clusters, "height_uncertainty_m": round(float(np.sqrt(s.P[2, 2])), 1)}

    # --- chat loop -----------------------------------------------------------------------
    async def chat(self, user, tools=None, max_rounds=4, context_numbers=()):
        """Run Nemotron with tools until it answers. Returns (answer, tool_log, allowed_numbers)."""
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        specs = self.tool_specs(tools)
        log, allowed = [], list(context_numbers)
        for _ in range(max_rounds):
            r = await asyncio.wait_for(asyncio.to_thread(
                self.client.chat.completions.create, model=self.cfg.llm_model, messages=msgs, tools=specs or None,
                temperature=0.2), self.cfg.llm_timeout_s)
            m = r.choices[0].message
            if not m.tool_calls:
                return (m.content or "").strip(), log, allowed
            msgs.append({"role": "assistant", "content": m.content or "", "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in m.tool_calls]})
            for c in m.tool_calls:
                try:
                    args = json.loads(c.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                try:
                    res = await self.call_tool(c.function.name, args)
                except Exception as e:                      # a broken tool call must not crash the layer
                    res = {"error": f"{type(e).__name__}: {e}"}
                log.append({"tool": c.function.name, "args": args, "result": res})
                _collect_numbers(res, allowed)
                _collect_numbers(args, allowed)
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(res, default=float)[:3000]})
        return "", log, allowed

    def _audit(self, kind, **kw):
        if self.log is not None:
            self.log.send({"kind": kind, "wall": time.time(), **kw})

    # --- jobs ----------------------------------------------------------------------------
    def known_names(self):
        """Every shelter, site and group name the site data knows (for the name check)."""
        names = {x["name"] for x in self.core.site.list_shelters()} | {x["name"] for x in self.core.site.list_sites()}
        return names | {g["name"] for g in self.core.site.d["groups"]} if hasattr(self.core.site, "d") else names

    async def personalise(self, alert, group):
        """A follow-up for one group after the core's alert: which shelter and how long they have.

        The facts (the group's shelter outside the danger zone, from the site-data MCP tool, and the alert's
        numbers) are fetched first and given to Nemotron, which only words the message. It is used only if
        every number comes from the facts or a tool result, every shelter or site it names exists, and it
        names the assigned shelter. Otherwise None (the core's message stands)."""
        nums = {k: v for k, v in alert["numbers"].items() if v is not None}
        site = next((x for x in self.core.site.list_sites() if x["name"] == nums.get("site")), None)
        t0 = time.time()
        try:
            args = {"x": group["x"], "y": group["y"]}
            if site:
                args.update(avoid_x=site["x"], avoid_y=site["y"], avoid_radius=nums.get("zone_radius_m", site["radius_m"]))
            shelter = await self.call_tool("nearest_shelter", args)
        except Exception as e:
            self._audit("personalise_failed", group=group["id"], error=f"site data: {type(e).__name__}: {e}")
            return None
        if "error" in shelter:
            return None
        arrive_s = nums.get("arrive_s")
        if arrive_s is not None and shelter["walk_min"] * 60 > arrive_s:
            # not enough time to reach it: never suggest moving (the core's "take cover now" stands)
            self._audit("personalise_skipped", group=group["id"], reason="shelter too far for the time left",
                        walk_s=round(shelter["walk_min"] * 60), arrive_s=arrive_s)
            return None
        facts = {"drone_heading_for": nums.get("site"), "seconds_until_drone_arrives": arrive_s,
                 "drone_height_m": nums.get("height"), "shelter": shelter["name"],
                 "shelter_distance_m": round(shelter["distance_m"]), "walk_minutes": max(1, round(shelter["walk_min"]))}
        by_unit = {"s": [arrive_s] if arrive_s is not None else [], "min": [facts["walk_minutes"]],
                   "m": [facts["shelter_distance_m"], nums.get("height", 0)]}
        prompt = (f"Alert already sent to {group['name']} ({group['kind']}, {group['size']} people): \"{alert['text']}\"\n"
                  f"Facts: {json.dumps(facts)}\n"
                  f"Write ONE short follow-up message for them (under 35 words): tell them to go to {shelter['name']}, "
                  f"how far it is and how long it takes, and how long until the drone arrives. "
                  f"Use only the facts. Reply with the message text only.")
        allowed = _collect_numbers(facts, [])
        try:
            text, log, allowed = await self.chat(prompt, ["nearest_shelter", "get_danger_zone"], context_numbers=allowed)
        except (asyncio.TimeoutError, Exception) as e:
            self._audit("personalise_failed", group=group["id"], error=f"{type(e).__name__}: {e}", seconds=time.time() - t0)
            return None
        text = text.strip().strip('"').strip()
        bad = check_numbers(text, allowed) + check_units(text, by_unit)
        names = self.unknown_names(text)
        ok = bool(text) and not bad and not names and shelter["name"].lower() in text.lower() and len(text.split()) <= 60
        self._audit("personalise", group=group["id"], text=text, facts=facts, tools=log, unsupported_numbers=bad,
                    unknown_names=names, used=ok, seconds=round(time.time() - t0, 2))
        if not ok:
            return None
        return {"t": alert["t"], "tier": alert["tier"] + "_followup", "to": group["id"], "text": text,
                "source": "nemotron", "seconds": round(time.time() - t0, 2)}

    def unknown_names(self, text):
        """Shelter/site-like names in the text ("Shelter X", "X Shelter", "Site X") that the site data does not have."""
        known = {n.lower() for n in self.known_names()}
        found = re.findall(r"\b(?:[Ss]helter\s+[A-Z0-9]\w*|[A-Z]\w+\s+[Ss]helter|[Ss]ite\s+[A-Z0-9]\w*)", text)
        return [f for f in found if f.lower() not in known]

    async def answer(self, question):
        """Operator question -> (answer or None, tool log). The current tracks and their danger zones are
        fetched first and given as facts; the answer is rejected if it uses a number or a name the facts
        and tool results do not support."""
        t0 = time.time()
        facts = {"tracks": [self._track_view(t) for t in self.core.latest],
                 "danger": [{"track_id": t, "sites": self._danger_view(t)["sites"]} for t in self.core.latest]}
        prompt = f"Facts (current system state): {json.dumps(facts)}\nOperator question: {question}\nAnswer from the facts only."
        try:
            text, log, allowed = await self.chat(prompt, context_numbers=_collect_numbers(facts, []))
        except (asyncio.TimeoutError, Exception) as e:
            self._audit("answer_failed", question=question, error=f"{type(e).__name__}: {e}")
            return None, []
        bad = check_numbers(text, allowed)
        names = self.unknown_names(text)
        self._audit("answer", question=question, text=text, tools=log, unsupported_numbers=bad, unknown_names=names,
                    seconds=round(time.time() - t0, 2))
        return (text if text and not bad and not names else None), log
