"""Border monitor: fixed-logic border events from the tracks (Border watch screen, terra/border.py).

The border is given as polylines in local metres with the UAE on the left. For each confirmed track
(at least COVER_UPDATES updates, not rated a ground source) it takes the signed distance to the border
(positive inside the UAE) and its rate of change, and sends to operators:

  border_watch   the track is beyond the border, within WATCH_M of it, or closing on it so that it would
                 cross within WATCH_S                                                   (once per track)
  border_cross   a track that was confirmed beyond the border is now inside by more than twice its
                 horizontal position sd (and at least MIN_INSIDE_M)                      (once per track)

A track first confirmed inside the UAE is never reported as a crossing: it may have started there.

One drone's track can break into several (fragmentation), and each piece would report again. So a
sector repeats an event of the same kind only for a track more than SAME_M from the last reported one, or
REPEAT_S after that one was last seen: within that, it is taken to be the same intruder.
"""
import math

import numpy as np

from alerting.core import COVER_UPDATES, OPERATORS

WATCH_M = 1500.0
WATCH_S = 120.0
MIN_INSIDE_M = 30.0
REPEAT_S = 60.0
SAME_M = 1500.0
NEIGHBOURS = {"SAU": "Saudi", "OMN": "Omani"}


class BorderMonitor:
    def __init__(self, lines, send, rate=None):
        """lines: [{"sector": id, "xy": [[x, y], ...], "neighbour": "SAU" | "OMN"}], UAE on the left;
        send(t, tier, to, text, **numbers) delivers an alert (AlertCore._send); rate(snapshot) -> (Rating, speed)."""
        self.segs = []
        for ln in lines:
            a = np.asarray(ln["xy"], float)
            for i in range(len(a) - 1):
                if np.hypot(*(a[i + 1] - a[i])) > 0:
                    self.segs.append((a[i], a[i + 1], ln["sector"], ln.get("neighbour", "")))
        self.send, self.rate = send, rate
        self.state = {}
        self.last = {}                          # (sector, tier) -> (t, x, y) of the last event sent

    def _repeat(self, t, sector, tier, x, y):
        """True if this would repeat a recent event of the same kind in the sector (the same intruder)."""
        prev = self.last.get((sector, tier))
        return prev is not None and t - prev[0] < REPEAT_S and math.hypot(x - prev[1], y - prev[2]) < SAME_M

    def signed_distance(self, x, y):
        """(signed distance m, positive inside the UAE; sector id; neighbour) to the nearest border segment."""
        p = np.array([x, y], float)
        best = None
        for a, b, sec, nb in self.segs:
            d = b - a
            u = float(np.clip((p - a) @ d / (d @ d), 0.0, 1.0))
            q = a + u * d
            r = float(np.hypot(*(p - q)))
            if best is None or r < best[0] - 1e-9:
                side = d[0] * (p - a)[1] - d[1] * (p - a)[0]          # > 0: left of the segment = UAE side
                best = (r, math.copysign(r, side) if r > 0 else 0.0, sec, nb, d / math.sqrt(d @ d))
        return best[1], best[2], best[3], best[4]

    def update(self, t, snapshots):
        out = []
        for tr in snapshots:
            if tr.n_upd < COVER_UPDATES:
                continue
            if self.rate is not None and self.rate(tr)[0].level == "ground":
                continue
            x, y, z = (float(v) for v in tr.state[:3])
            vx, vy = float(tr.state[3]), float(tr.state[4])
            sd, sector, nb, tan = self.signed_distance(x, y)
            normal = np.array([-tan[1], tan[0]])                    # into the UAE
            closing = float(np.array([vx, vy]) @ normal)            # m/s towards the UAE side
            sig = float(np.sqrt(max(tr.P[0, 0], tr.P[1, 1])))
            st = self.state.setdefault(tr.id, {"outside": False, "watched": False, "crossed": False})
            speed = math.hypot(vx, vy)
            hdg = (math.degrees(math.atan2(vx, vy)) + 360) % 360
            side = NEIGHBOURS.get(nb, "far")
            for tier, key in (("border_watch", "watched"), ("border_cross", "crossed")):
                if st[key]:                             # still the reported intruder: keep its window open
                    self.last[(sector, tier)] = (t, x, y)
            if sd < -max(2 * sig, MIN_INSIDE_M):
                st["outside"] = True
                eta = -sd / closing if closing > 0.5 else None
                if not st["watched"] and (-sd <= WATCH_M or (eta is not None and eta <= WATCH_S)):
                    st["watched"] = True
                    if self._repeat(t, sector, "border_watch", x, y):
                        continue
                    self.last[(sector, "border_watch")] = (t, x, y)
                    when = f", crossing in about {eta:.0f} s" if eta is not None and eta <= WATCH_S else ""
                    out.append(self.send(t, "border_watch", OPERATORS,
                                         f"BORDER {sector}: drone track {tr.id} on the {side} side, {-sd:.0f} m out, "
                                         f"heading {hdg:03.0f} at {speed:.0f} m/s{when}.",
                                         track=tr.id, sector=sector, distance_m=round(-sd), speed_m_s=round(speed, 1),
                                         heading_deg=round(hdg), eta_s=None if eta is None else round(eta)))
            elif sd > max(2 * sig, MIN_INSIDE_M) and st["outside"] and not st["crossed"]:
                st["crossed"] = True
                if self._repeat(t, sector, "border_cross", x, y):
                    continue
                self.last[(sector, "border_cross")] = (t, x, y)
                out.append(self.send(t, "border_cross", OPERATORS,
                                     f"BORDER CROSSING {sector}: drone track {tr.id} is {sd:.0f} m inside the UAE "
                                     f"(+-{sig:.0f} m), heading {hdg:03.0f} at {speed:.0f} m/s, "
                                     f"about {z:.0f} m above the ground.",
                                     track=tr.id, sector=sector, inside_m=round(sd), sd_m=round(sig),
                                     speed_m_s=round(speed, 1), heading_deg=round(hdg), height_m=round(z)))
        return out
