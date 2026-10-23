"""Tabletop exercise: one drone approaches the example site from 1.4 km, so every tier fires
(on course -> evacuation approval -> evacuate -> take cover -> all clear) and Nemotron writes the
follow-ups and answers operator questions.

    python -m alerting.demo [--no-llm] [--approve-after 10]

The track is scripted (straight approach at 10 m/s, 60 m up, a turn away after the pass) with a fixed
covariance, i.e. what the Kalman filter would report with good coverage. It exercises the alerting
logic, not the acoustics; for acoustics-driven runs use alerting.replay on the simulated scenes.
"""
import argparse
import asyncio
import json

import numpy as np

from alerting.config import Config
from alerting.core import AlertCore, TrackSnapshot
from alerting.delivery import ApprovalQueue, Delivery, LogSink, PrintSink, cot_event
from alerting.sitedata import LocalSiteData

QUESTIONS = {40.0: "What is the drone doing and which groups should move where?",
             110.0: "How long until the drone reaches the Village, and how sure are we?"}


def path(t):
    """x, y, z, vx, vy, vz: from (1400, 120) west at 10 m/s at 60 m; turns south after passing x = 150."""
    if t < 125:
        return np.array([1400 - 10 * t, 120.0, 60.0, -10.0, 0.0, 0.0])
    s = t - 125
    return np.array([150.0, 120 - 10 * s, 60.0, 0.0, -10.0, 0.0])


async def run(use_llm, approve_after, out):
    cfg = Config()
    site = LocalSiteData(cfg.site_file)
    log = LogSink(out)
    core = AlertCore(site, Delivery([log, PrintSink()]), ApprovalQueue(auto_after_s=approve_after))
    P = np.diag([4.0 ** 2] * 3 + [0.8 ** 2] * 3)
    agent = None
    if use_llm:
        from alerting.agent import NemotronLayer
        agent = await NemotronLayer(core, cfg, log=log).__aenter__()
    try:
        for i in range(int(160 / 0.1)):
            t = i * 0.1
            snaps = [TrackSnapshot(1, path(t), P, 30 + i, 3, 0.9, 20.0 + t)] if t < 150 else []
            events = core.tick(t, snaps)
            if agent is not None:
                for ev in events:
                    if ev["tier"] in ("take_cover", "evacuate") and ev["to"] != "operators":
                        r = await agent.personalise(ev, site.get_group(ev["to"]))
                        if r is not None:
                            core.delivery.send(r)
                if round(t, 1) in QUESTIONS:
                    q = QUESTIONS[round(t, 1)]
                    a, _ = await agent.answer(q)
                    core.delivery.send({"t": round(t, 1), "tier": "operator_qa", "to": "operators",
                                        "text": f"Q: {q} A: {a or '(answer withheld: failed the fact check)'}", "source": "nemotron"})
            if round(t, 1) == 60.0:
                core.delivery.send({"t": 60.0, "tier": "cot", "to": "ATAK", "text": cot_event(1, *path(60)[:3], site.origin_latlon())})
    finally:
        if agent is not None:
            await agent.__aexit__(None, None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--approve-after", type=float, default=10.0, help="seconds before the replay operator approves")
    ap.add_argument("--out", default="results/system_demo/tabletop.jsonl")
    a = ap.parse_args()
    from pathlib import Path
    Path(a.out).unlink(missing_ok=True)
    asyncio.run(run(not a.no_llm, a.approve_after, a.out))
    tiers = [json.loads(line).get("tier") for line in open(a.out)]
    print({k: tiers.count(k) for k in dict.fromkeys(tiers) if k})


if __name__ == "__main__":
    main()
