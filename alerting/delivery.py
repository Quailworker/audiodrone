"""Getting alerts to people (design.md 13.4): a message log, optional webhook, ATAK Cursor-on-Target.

Real channels (radio nets, SMS gateways, sirens, phone push) plug in as sinks with a send(msg) method.
Everything sent is logged as JSON lines, so the whole alert history can be replayed and audited.
"""
import json
import time
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.sax.saxutils import quoteattr

from alerting.sitedata import to_latlon


class LogSink:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, msg):
        with self.path.open("a") as f:
            f.write(json.dumps(msg, default=float) + "\n")


class PrintSink:
    def send(self, msg):
        print(f"[{msg.get('t', 0):7.1f}s] {msg.get('tier', ''):10s} -> {msg.get('to', '')}: {msg.get('text', '')}", flush=True)


class WebhookSink:
    """POST each message as JSON (e.g. to an ntfy topic, a phone app backend or an SMS gateway)."""

    def __init__(self, url, timeout=3.0):
        self.url, self.timeout = url, timeout

    def send(self, msg):
        req = urllib.request.Request(self.url, json.dumps(msg, default=float).encode(), {"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=self.timeout).read()
        except OSError as e:                       # delivery failure must never stop the alert core
            msg.setdefault("delivery_errors", []).append(str(e))


class Delivery:
    def __init__(self, sinks):
        self.sinks = list(sinks)

    def send(self, msg):
        msg.setdefault("sent_wall", time.time())
        for s in self.sinks:
            s.send(msg)
        return msg


class ApprovalQueue:
    """Evacuation orders wait here for an operator (D-35). `auto_after_s` approves automatically in replays."""

    def __init__(self, auto_after_s=None):
        self.pending, self.decided, self.auto_after_s = {}, {}, auto_after_s

    def request(self, order, t):
        oid = order.get("id") or uuid.uuid4().hex[:8]
        self.pending[oid] = {**order, "id": oid, "requested_t": t}
        return oid

    def decide(self, oid, approved, t, by="operator"):
        o = self.pending.pop(oid, None)
        if o is not None:
            self.decided[oid] = {**o, "approved": approved, "decided_t": t, "by": by}
        return self.decided.get(oid)

    def tick(self, t):
        if self.auto_after_s is None:
            return []
        due = [oid for oid, o in self.pending.items() if t - o["requested_t"] >= self.auto_after_s]
        return [self.decide(oid, True, t, by="auto (replay)") for oid in due]


def cot_event(track_id, x, y, z, origin, stale_s=10, how="m-g", likely=True):
    """Cursor-on-Target XML for one track (ATAK). Type a-h-A-M-F-Q: hostile air, military, UAV
    (a-u-A, unknown air, when the rating is only "possible")."""
    lat, lon = to_latlon(origin, x, y)
    now = datetime.now(timezone.utc)
    fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
    typ = "a-h-A-M-F-Q" if likely else "a-u-A"
    return (f'<event version="2.0" uid={quoteattr(f"audiodrone-track-{track_id}")} type="{typ}" how="{how}" '
            f'time="{now.strftime(fmt)}" start="{now.strftime(fmt)}" stale="{(now + timedelta(seconds=stale_s)).strftime(fmt)}">'
            f'<point lat="{lat:.7f}" lon="{lon:.7f}" hae="{z:.1f}" ce="15" le="10"/>'
            f'<detail><contact callsign={quoteattr(f"Drone {track_id}")}/></detail></event>')
