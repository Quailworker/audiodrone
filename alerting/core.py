"""Fixed-logic alert core (design.md 13.4, D-34, D-35). No language model in this path.

Every 0.1 s it takes the current tracks (Kalman state + covariance + CNN evidence) and the CNN
detections that have no track yet, rates them (D-32/D-38), assesses each protected site (13.3) and
sends tiered alerts with template messages:

  advisory    possible drone (CNN only, no usable position, or rating "possible")  -> operators
  take cover  drone likely and a site threatened                                    -> everyone in the zone,
                                                                                       automatically
  on course   drone likely and its predicted path runs through a site, but the     -> operators
              prediction is not yet confident enough for automatic take-cover
  evacuate    drone likely, site threatened or on course, and time to reach a      -> needs operator
              shelter outside the zone before arrival                                  approval, unless the
                                                                                       site is threatened and
                                                                                       arrival is too soon
                                                                                       to wait for one
  all clear   a threatened track is gone or no longer threatening for CLEAR_S

Messages are re-sent only on escalation or every REPEAT_S, so people are not flooded. Every alert
carries the numbers it was built from, which the Nemotron layer may reuse (and nothing else).
"""
from dataclasses import dataclass, field

import numpy as np

from alerting import danger
from alerting.rating import Evidence, rate

TIERS = ["log", "advisory", "take_cover", "evacuate"]
REPEAT_S = 30.0
CLEAR_S = 10.0
APPROVAL_BUDGET_S = 60.0          # arrival sooner than this: evacuate without waiting for approval
CONFIRM_UPDATES = 20              # Kalman updates (~2 s with several clusters) before evacuation is offered
COVER_UPDATES = 10                # updates before a track may trigger take-cover (a new track's P is still the prior)
SIGMA_ALERT_M = 30.0              # horizontal position sd above which a threat is only an operator advisory
OPERATORS = "operators"


@dataclass
class TrackSnapshot:
    id: int
    state: np.ndarray             # x, y, z, vx, vy, vz (m, m/s)
    P: np.ndarray                 # 6x6 covariance
    n_upd: int
    n_clusters: int
    p_cnn: float
    detected_s: float
    duty: float = 1.0


@dataclass
class Detection:
    """The CNN fired at a cluster, but there is no track (yet)."""
    cluster: int
    p_cnn: float
    detected_s: float
    duty: float = 1.0


@dataclass
class _State:
    tier: str = "log"
    last_sent: float = -1e9
    last_threat: float = -1e9
    sites: set = field(default_factory=set)
    approval: str | None = None


class AlertCore:
    def __init__(self, site, delivery, approvals, cluster_xy=None):
        self.site, self.delivery, self.approvals = site, delivery, approvals
        self.sites = site.list_sites()
        self.cluster_xy = cluster_xy
        self.state: dict = {}
        self.events = []
        self.latest: dict = {}            # track id -> latest snapshot, rating and threats (for the Nemotron tools)

    # --- helpers -------------------------------------------------------------------------
    def _send(self, t, tier, to, text, **numbers):
        msg = {"t": round(t, 2), "tier": tier, "to": to, "text": text, "source": "core", "numbers": numbers}
        self.delivery.send(msg)
        self.events.append(msg)
        return msg

    def rate_track(self, tr: TrackSnapshot):
        speed = float(np.linalg.norm(tr.state[3:]))
        return rate(Evidence(tr.p_cnn, tr.detected_s, tr.duty, float(tr.state[2]), float(np.sqrt(tr.P[2, 2])),
                             speed, tr.n_clusters)), speed

    # --- one tick ------------------------------------------------------------------------
    def tick(self, t, tracks, detections=()):
        out = []
        seen = set()
        for tr in tracks:
            seen.add(tr.id)
            out += self._track(t, tr)
        for det in detections:
            out += self._detection(t, det)
        for tid in [k for k in self.latest if k not in seen]:
            self.latest.pop(tid)
        for tid, st in list(self.state.items()):
            if isinstance(tid, int) and tid not in seen and st.tier in ("take_cover", "evacuate"):
                out.append(self._send(t, "all_clear", OPERATORS, f"Drone {tid} lost. Stand by; alerts resume if it is heard again.", track=tid))
                st.tier = "log"
        for o in self.approvals.tick(t):
            if o and o["approved"]:
                out += self._evacuate(t, o)
        return out

    def _detection(self, t, det):
        st = self.state.setdefault("untracked", _State())         # one advisory stream, not one per cluster
        r = rate(Evidence(det.p_cnn, det.detected_s, det.duty))
        if det.detected_s < 0.5 or t - st.last_sent < REPEAT_S:
            return []
        st.last_sent, st.tier = t, "advisory"
        where = f" near cluster {det.cluster}" if self.cluster_xy is None else \
            f" near cluster {det.cluster} ({self.cluster_xy[det.cluster][0]:.0f} m E, {self.cluster_xy[det.cluster][1]:.0f} m N)"
        return [self._send(t, "advisory", OPERATORS, f"Possible drone heard{where}. No position yet.",
                           cluster=det.cluster, p_cnn=round(det.p_cnn, 2), rating=r.level)]

    def _track(self, t, tr):
        st = self.state.setdefault(tr.id, _State())
        r, speed = self.rate_track(tr)
        x, y, z = (float(v) for v in tr.state[:3])
        nums = dict(track=tr.id, x=round(x), y=round(y), height=round(z), speed=round(speed, 1),
                    likelihood=None if r.likelihood is None else round(r.likelihood, 2), rating=r.level)
        self.latest[tr.id] = {"t": t, "snapshot": tr, "rating": r, "speed": speed, "threats": []}
        out = []
        if r.level == "ground":
            if st.tier == "log" and t - st.last_sent >= REPEAT_S:
                st.last_sent = t
                self.events.append({"t": round(t, 2), "tier": "log", "to": "log", "source": "core",
                                    "text": f"Track {tr.id} rated ground source ({', '.join(r.reasons)}).", "numbers": nums})
            return out
        if r.level == "possible":
            if TIERS.index(st.tier) < 1 or t - st.last_sent >= REPEAT_S:
                st.tier, st.last_sent = max(st.tier, "advisory", key=TIERS.index), t
                out.append(self._send(t, "advisory", OPERATORS,
                                      f"Possible drone, track {tr.id}: {x:.0f} m E, {y:.0f} m N, about {z:.0f} m up, "
                                      f"{speed:.0f} m/s. Not yet rated likely.", **nums))
            return out
        # likely: assess every protected site
        threats = [danger.assess(tr.state, tr.P, (s["x"], s["y"]), s["radius_m"], s["name"]) for s in self.sites]
        self.latest[tr.id]["threats"] = threats
        hit = sorted([th for th in threats if th.threatened], key=lambda th: th.tau_arrive)
        course = sorted([th for th in threats if th.on_course and not th.threatened], key=lambda th: th.tau_min)
        if not hit and course:
            if tr.n_upd >= COVER_UPDATES and danger.sigma_at(tr.P, 0.0) <= SIGMA_ALERT_M:
                return out + self._on_course(t, tr, st, course[0], nums)
            if TIERS.index(st.tier) < 1 or t - st.last_sent >= REPEAT_S:
                st.tier, st.last_sent = max(st.tier, "advisory", key=TIERS.index), t
                out.append(self._send(t, "advisory", OPERATORS,
                                      f"Drone likely, track {tr.id}, possibly on course for {course[0].site}; position "
                                      f"still uncertain.", site=course[0].site, **nums))
            return out
        if not hit:
            if st.tier in ("take_cover", "evacuate") and t - st.last_threat >= CLEAR_S:
                st.tier = "advisory"
                out.append(self._send(t, "all_clear", OPERATORS, f"Drone {tr.id} no longer heading for a protected site. Stay alert.", **nums))
            elif TIERS.index(st.tier) < 1 or t - st.last_sent >= REPEAT_S:
                st.tier, st.last_sent = "advisory", t
                out.append(self._send(t, "advisory", OPERATORS,
                                      f"Drone likely, track {tr.id}: {x:.0f} m E, {y:.0f} m N, {z:.0f} m up, {speed:.0f} m/s. "
                                      f"No protected site on its course.", **nums))
            return out
        th = hit[0]
        sigma_now = danger.sigma_at(tr.P, 0.0)
        if tr.n_upd < COVER_UPDATES or sigma_now > SIGMA_ALERT_M:
            if TIERS.index(st.tier) < 1 or t - st.last_sent >= REPEAT_S:
                st.tier, st.last_sent = max(st.tier, "advisory", key=TIERS.index), t
                out.append(self._send(t, "advisory", OPERATORS,
                                      f"Drone likely, track {tr.id}, possibly heading for {th.site}; position still uncertain "
                                      f"(+-{sigma_now:.0f} m). Take-cover follows when the track firms up.",
                                      site=th.site, sigma_m=round(sigma_now), **nums))
            return out
        st.last_threat = t
        site = next(s for s in self.sites if s["name"] == th.site)
        zone_r = site["radius_m"] + min(danger.K_SIGMA * th.sigma_at_arrival, site["radius_m"])   # people search radius, at most 2R
        nums.update(site=th.site, arrive_s=round(th.tau_arrive), closest_m=round(th.d_min), zone_radius_m=round(zone_r),
                    probability=round(th.q_max, 2))
        escalate = TIERS.index(st.tier) < TIERS.index("take_cover") or th.site not in st.sites
        if escalate or t - st.last_sent >= REPEAT_S:
            groups = self.site.people_in_zone(site["x"], site["y"], zone_r)
            st.tier = max(st.tier, "take_cover", key=TIERS.index)
            st.last_sent = t
            st.sites.add(th.site)
            when = "over it now" if th.tau_arrive < 1 else f"about {th.tau_arrive:.0f} s away"
            text = f"TAKE COVER: drone heading for {th.site}, {when}, {z:.0f} m up. Get under hard cover now."
            for g in groups:
                out.append(self._send(t, "take_cover", g["id"], text, group=g["name"], **nums))
            sent = f"Take-cover sent to {len(groups)} group(s) at {th.site}" if groups else \
                f"TAKE COVER at {th.site} (no groups on record there)"
            out.append(self._send(t, "take_cover", OPERATORS,
                                  f"{sent}: drone {tr.id}, {when}, "
                                  f"{z:.0f} m up, {speed:.0f} m/s, closest pass {th.d_min:.0f} m (+-{sigma_now:.0f} m).",
                                  groups=[g["id"] for g in groups], **nums))
            out += self._offer_evacuation(t, tr, st, site, th.site, th.tau_arrive, zone_r, groups, nums, allow_auto=True)
        return out

    def _on_course(self, t, tr, st, th, nums):
        """Early warning: the predicted path runs through a site, prediction not yet confident."""
        out = []
        site = next(s for s in self.sites if s["name"] == th.site)
        zone_r = site["radius_m"] + min(danger.K_SIGMA * th.sigma_at_arrival, site["radius_m"])
        nums = {**nums, "site": th.site, "closest_s": round(th.tau_min), "closest_m": round(th.d_min),
                "probability": round(th.q_max, 2), "zone_radius_m": round(zone_r)}
        if th.site not in st.sites or t - st.last_sent >= REPEAT_S:
            st.sites.add(th.site)
            st.last_sent = t
            st.tier = max(st.tier, "advisory", key=TIERS.index)
            out.append(self._send(t, "on_course", OPERATORS,
                                  f"Drone {tr.id} is on course for {th.site}: closest pass {th.d_min:.0f} m in about "
                                  f"{th.tau_min:.0f} s if it holds course (not yet certain).", **nums))
            groups = self.site.people_in_zone(site["x"], site["y"], zone_r)
            out += self._offer_evacuation(t, tr, st, site, th.site, th.tau_min, zone_r, groups, nums, allow_auto=False)
        return out

    def _offer_evacuation(self, t, tr, st, site, site_name, tau, zone_r, groups, nums, allow_auto):
        if tr.n_upd < CONFIRM_UPDATES or st.approval is not None or not groups:
            return []
        plan = []
        for g in groups:
            sh = self.site.nearest_shelter(g["x"], g["y"], site["x"], site["y"], zone_r)
            if "error" in sh or sh["walk_min"] * 60 > tau:
                continue                                   # cannot get there in time: take cover in place
            plan.append({"group": g["id"], "group_name": g["name"], "shelter": sh["name"],
                         "distance_m": sh["distance_m"], "walk_min": sh["walk_min"]})
        if not plan:
            return []
        order = {"track": tr.id, "site": site_name, "arrive_s": round(tau), "plan": plan, "t": t}
        if allow_auto and tau < APPROVAL_BUDGET_S:
            st.approval = "auto"
            return self._evacuate(t, {**order, "approved": True, "by": "auto (arrival too soon to wait)"})
        st.approval = self.approvals.request(order, t)
        return [self._send(t, "approval", OPERATORS,
                           f"APPROVE EVACUATION of {len(plan)} group(s) from {site_name}? Drone {tr.id} about "
                           f"{tau:.0f} s away. Order {st.approval}.", order=st.approval, **nums)]

    def _evacuate(self, t, order):
        st = self.state.get(order["track"])
        if st is not None:
            st.tier = "evacuate"
        out = []
        for p in order["plan"]:
            out.append(self._send(t, "evacuate", p["group"],
                                  f"EVACUATE to {p['shelter']}, {p['distance_m']:.0f} m away (about {max(1, round(p['walk_min']))} min on foot). "
                                  f"Drone heading for {order['site']}, about {order['arrive_s']:.0f} s out when ordered. Avoid {order['site']}.",
                                  track=order["track"], site=order["site"], shelter=p["shelter"], distance_m=p["distance_m"],
                                  walk_min=p["walk_min"], arrive_s=order["arrive_s"], approved_by=order.get("by")))
        return out
