"""End-to-end replay of a localisation scene: directions -> tracker -> rating -> danger zone -> alerts
-> Nemotron follow-ups (design.md 13.3-13.4).

    python -m alerting.replay --scene data/loc_v1/flat_ground_100/L20000123 [--method bf_comb2k] [--no-llm]
    python -m alerting.replay --pick 3            # pick scenes whose drone passes a protected site

Detection is the oracle stand-in used in Phase 1: a cluster "hears" the drone when it is audible there by
the CNN label rule (loc/evaluate.py), with p_cnn = 0.9. The CNN itself needs 32 kHz audio, which the
localisation scenes do not keep.

Writes results/system_demo/<scene>.jsonl (every alert, follow-up, approval and Nemotron audit record)
and prints the alert timeline.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np

from alerting.config import Config
from alerting.core import AlertCore, Detection, TrackSnapshot
from alerting.delivery import ApprovalQueue, Delivery, LogSink, PrintSink
from alerting.sitedata import LocalSiteData
from loc.evaluate import audible, load_scene
from loc.track import Track, Tracker

COH_GATE = 0.5
P_CNN = 0.9


def scene_inputs(stem, method):
    meta, truth, doa = load_scene(stem)
    if doa is None or f"{method}/c0/u" not in doa:
        raise SystemExit(f"{stem}: no {method} directions (run loc.run_doa first)")
    tf = truth["frame_t"]
    centres = np.asarray(meta["centres"])
    K = len(centres)
    aud = np.stack([audible(truth, meta, k) for k in range(K)])
    ok = aud & np.stack([doa[f"{method}/c{k}/coh"].astype(float) >= COH_GATE for k in range(K)])
    meas = [[(k, doa[f"{method}/c{k}/u"][i].astype(float)) for k in range(K) if ok[k, i]] for i in range(len(tf))]
    return meta, truth, tf, centres, aud, meas


def run(stem, method="bf", cfg=None, use_llm=True, out_dir="results/system_demo", quiet=False, auto_approve_s=15.0):
    return asyncio.run(_run(stem, method, cfg or Config(), use_llm, out_dir, quiet, auto_approve_s))


async def _run(stem, method, cfg, use_llm, out_dir, quiet, auto_approve_s):
    meta, truth, tf, centres, aud, meas = scene_inputs(stem, method)
    out = Path(out_dir) / f"{Path(stem).name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    log = LogSink(out)
    sinks = [log] if quiet else [log, PrintSink()]
    site = LocalSiteData(cfg.site_file)
    core = AlertCore(site, Delivery(sinks), ApprovalQueue(auto_after_s=auto_approve_s), cluster_xy=centres[:, :2])
    Track._next = 0
    trk = Tracker(centres, flat=meta["cluster"] == "flat", sigma_u=0.04)
    heard_since, history = None, []
    tasks, followups, core_wall = [], [], []
    agent = None
    if use_llm:
        from alerting.agent import NemotronLayer
        agent = await NemotronLayer(core, cfg, log=log).__aenter__()
    try:
        for i, t in enumerate(tf):
            tracks = trk.step(t, meas[i])
            heard = bool(aud[:, i].any())
            heard_since = (heard_since if heard_since is not None else t) if heard else None
            history.append(heard)
            duty = float(np.mean(history[-50:]))
            detected_s = 0.0 if heard_since is None else t - heard_since + 0.1
            n_aud = int(aud[:, i].sum())
            snaps = [TrackSnapshot(tr.id, tr.s.copy(), tr.P.copy(), tr.n_upd, n_aud, P_CNN if heard else 0.3,
                                   detected_s, duty) for tr in tracks if tr.n_upd >= 5]
            dets = [] if snaps or not heard else [Detection(int(np.argmax(aud[:, i])), P_CNN, detected_s, duty)]
            w0 = time.perf_counter()
            events = core.tick(float(t), snaps, dets)
            core_wall.append(time.perf_counter() - w0)
            if agent is not None:
                for ev in events:
                    if ev["tier"] in ("take_cover", "evacuate") and ev["to"] != "operators":
                        g = site.get_group(ev["to"])
                        tasks.append(asyncio.create_task(agent.personalise(ev, g)))
            if tasks:
                await asyncio.sleep(0)                     # let Nemotron work alongside the core
        if tasks:
            for r in await asyncio.gather(*tasks):
                if r is not None:
                    followups.append(r)
                    core.delivery.send(r)
        answer = None
        if agent is not None and core.latest:
            answer, _, unverified = await agent.answer("Summarise for the operator in under 60 words: which tracks exist, "
                                                       "how likely each is a drone, and which site is threatened and when.")
            if answer and unverified:
                answer += f" [unverified: {', '.join(unverified)}]"
    finally:
        if agent is not None:
            await agent.__aexit__(None, None, None)
    summary = {"scene": Path(stem).name, "method": method, "events": len(core.events),
               "by_tier": {k: sum(e["tier"] == k for e in core.events) for k in
                           ("advisory", "on_course", "take_cover", "approval", "evacuate", "all_clear", "log")},
               "nemotron_followups": len(followups), "nemotron_attempts": len(tasks),
               "core_ms_per_tick_p99": round(1000 * float(np.percentile(core_wall, 99)), 2),
               "first_alert_t": next((e["t"] for e in core.events if e["tier"] != "log"), None),
               "first_take_cover_t": next((e["t"] for e in core.events if e["tier"] == "take_cover"), None),
               "operator_summary": answer}
    log.send({"kind": "summary", **summary})
    return summary, core.events, followups


def pick_scenes(config="flat_ground_100", method="bf", n=3, site_file=None):
    """Single-drone scenes whose drone passes within a protected site's radius while tracked."""
    site = LocalSiteData(site_file) if site_file else LocalSiteData()
    out = []
    for js in sorted(Path("data/loc_v1", config).glob("L*.doa.npz")):
        stem = str(js)[:-8]
        meta, truth, doa = load_scene(stem)
        if doa is None or f"{method}/c0/u" not in doa or sum(s["category"].startswith("drone") for s in meta["sources"]) != 1:
            continue
        p = truth["d0/p"][truth["d0/t"] >= 0]
        near = any(np.min(np.hypot(p[:, 0] - s["x"], p[:, 1] - s["y"])) < s["radius_m"] for s in site.list_sites())
        aud = np.stack([audible(truth, meta, k) for k in range(len(meta["centres"]))])
        if near and (aud.sum(0) >= 2).mean() > 0.3:
            out.append(stem)
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene")
    ap.add_argument("--pick", type=int, default=0)
    ap.add_argument("--config", default="flat_ground_100")
    ap.add_argument("--method", default="bf")
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()
    scenes = [a.scene] if a.scene else pick_scenes(a.config, a.method, a.pick or 1)
    for s in scenes:
        summary, _, followups = run(s, a.method, use_llm=not a.no_llm)
        for f in followups:
            print(f"  nemotron -> {f['to']}: {f['text']}")
        print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
