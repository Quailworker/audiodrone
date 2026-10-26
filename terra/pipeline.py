"""Run the whole chain on a rendered terrain scenario and package the result for the tool.

    detection   the real CNN (model 1, B_logf) on each cluster's reference mic, 32 kHz
    pitch       the frequency tracker (loc/freqtrack.py) on each cluster's reference mic, 8 kHz   [chain v2]
    direction   the reflection-aware beamformer per cluster (loc/doa.py, 100-860 Hz); in chain v2 steered
                on the tracker's harmonic lines where it holds a tone (bf_trk), else the whole band
    tracking    the EKF over all clusters (loc/track.py) with the terrain as the ground; chain v2 adds the
                tracked pitches as Doppler measurements
    alerts      rating, danger zone and alert tiers (alerting/core.py), height above the local ground

scenario["chain"]: "v2" (default) or "v1" (the v0.9 chain: CNN + whole-band beamformer, directions only).
scenario["border"]: optional border lines in local metres, UAE on the left ([{"sector", "neighbour", "xy"}]);
adds border_watch / border_cross alerts (alerting/border.py).

A cluster's direction is used in a frame when its CNN says drone (p >= P_DET over the 0.1 s frame) and
the beamformer's coherence passes COH_GATE, the same gate as Phase 1.
"""
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from alerting.border import BorderMonitor
from alerting.core import AlertCore, Detection, TrackSnapshot
from alerting.delivery import ApprovalQueue, Delivery, LogSink
from alerting.sitedata import LocalSiteData
from cnn.evaluate import predict
from cnn.model import CLASSES, DroneCNN14
from loc.doa import BeamformerDOA, angle_deg
from loc.evaluate import audible
from loc.freqtrack import HarmonicTracker
from loc.render import truth as make_truth
from loc.run_doa import tracker_weights
from loc.track import TRACKER_V12, Track, Tracker

from .scenario import FRAME_HZ, FS_OUT, render

MODEL = Path(__file__).resolve().parent.parent / "models" / "B_logf_best.pt"
P_DET = 0.5
COH_GATE = 0.5
DRONE_IDX = [CLASSES.index("drone_electric"), CLASSES.index("drone_engine")]


@lru_cache(maxsize=1)
def cnn(device):
    m = DroneCNN14("logf").to(device).eval()
    m.load_state_dict(torch.load(MODEL, map_location=device))
    return m


class SiteDict(LocalSiteData):
    def __init__(self, d):
        self.d = d


def run(scn, out_dir, progress=print, workers=6):
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress("rendering sound at every cluster")
    shared, clusters = render(scn, workers=workers, progress=progress)
    terrain, centres, duration = shared["terrain"], shared["centres"], shared["duration"]
    tf = np.arange(0, duration, 1 / FRAME_HZ) + 0.5 / FRAME_HZ
    K, F = len(centres), len(tf)
    t_render = time.time() - t0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    progress("CNN detection on each cluster")
    p_cnn = np.zeros((K, F), np.float32)
    model = cnn(device)
    for k, _x8, ref, _lab, _scr in clusters:
        probs = predict(model, ref, device)[:, DRONE_IDX].max(1)                       # 100 frames/s
        idx = np.clip((tf * 100).astype(int), 0, len(probs) - 1)
        p_cnn[k] = np.array([probs[max(i - 5, 0): i + 5].mean() for i in idx])
        sf.write(out_dir / f"cluster{k}.wav", ref[::2], 16000, subtype="PCM_16")      # to listen in the tool
    v2 = scn.get("chain", "v2") == "v2"
    doppler = v2 and scn.get("doppler", False)                    # opt-in: hurt tracking on the 120 flat scenes (results/chain_v2)
    pitch = np.full((K, F, 2), np.nan)
    if v2:
        progress("frequency tracking")
        ht = HarmonicTracker(FS_OUT)
        for k, x8, _ref, _lab, _scr in clusters:
            r = ht.run(x8[0], tf)
            pitch[k] = np.where(np.isfinite(r["f0_fine"]), r["f0_fine"], r["f0"])
    progress("direction finding")
    bf = BeamformerDOA(shared["offsets"], FS_OUT, shared["mount"])
    U = np.zeros((K, F, 3), np.float32)
    coh = np.zeros((K, F), np.float32)
    for k, x8, _ref, _lab, _scr in clusters:
        r = bf.run(x8, tf)
        U[k], coh[k] = r["u"], r["coh"]
        if v2:
            tone = np.isfinite(pitch[k, :, 0])
            if tone.any():
                w = tracker_weights({"c0/f0": pitch[k], "c0/f0_fine": pitch[k]}, 0, bf.base.f)
                rt = bf.run(x8, tf, chunk=4, weights=w)
                U[k, tone], coh[k, tone] = rt["u"][tone], rt["coh"][tone]
    del bf
    if device == "cuda":
        torch.cuda.empty_cache()

    progress("tracking and alerts")
    truth = make_truth(shared["tracks"], centres, [c[3] for c in clusters], duration)
    meta = {"sources": [{"category": s[0]} for s in shared["sources"]]}
    aud = np.stack([audible(truth, meta, k) for k in range(K)])
    det = p_cnn >= P_DET
    use = det & (coh >= COH_GATE)
    frames, events = track_and_alert(scn, terrain, centres, tf, U, use, det, p_cnn, pitch, doppler, out_dir / "alerts.jsonl")
    result = package(scn, shared, clusters, truth, tf, p_cnn, det, use, aud, U, coh, frames, events)
    result["pitch_hz"] = np.round(np.where(np.isfinite(pitch[..., 0]), pitch[..., 0], 0), 1).tolist()
    result["chain"] = "v2" if v2 else "v1"
    result["timing"] = {"render_s": round(t_render, 1), "total_s": round(time.time() - t0, 1)}
    progress("done")
    return result


# Tracker settings for this chain: the flat-cluster profile (loc/track.py TRACKER_V12: three-ray births, an
# 800 m height cap, and duplicate merging that adapts to whether clusters report one direction or several).
TRACKER_KW = {"sigma_u": 0.04, "birth_range": 4000.0, **TRACKER_V12}


def track_and_alert(scn, terrain, centres, tf, U, use, det, p_cnn, pitch, doppler, log_path, tracker_kw=None):
    """The EKF over all clusters' directions, then the alert core and the border monitor, frame by frame.
    Returns (frames for the result, the core's events)."""
    K = len(centres)
    Track._next = 0
    trk = Tracker(centres, flat=True, ground=terrain.height, **(TRACKER_KW if tracker_kw is None else tracker_kw))
    site = SiteDict({"sites": scn.get("sites", []), "shelters": scn.get("shelters", []),
                     "groups": scn.get("groups", []), "origin_latlon": list(scn.get("origin", [0, 0]))})
    Path(log_path).unlink(missing_ok=True)                                         # a rerun into the same folder starts afresh
    log = LogSink(log_path)
    core = AlertCore(site, Delivery([log]), ApprovalQueue(auto_after_s=15.0), cluster_xy=centres[:, :2])
    border = BorderMonitor(scn["border"], core._send, core.rate_track) if scn.get("border") else None
    heard_since, history, frames = None, [], []
    for i, t in enumerate(tf):
        meas = [(k, U[k, i].astype(float)) for k in range(K) if use[k, i]]
        freqs = {k: [f for f in pitch[k, i] if np.isfinite(f)] for k in range(K)} if doppler else None
        tracks = trk.step(t, meas, freqs)
        heard = bool(det[:, i].any())
        heard_since = (heard_since if heard_since is not None else t) if heard else None
        history.append(heard)
        duty = float(np.mean(history[-50:]))
        detected_s = 0.0 if heard_since is None else t - heard_since + 0.1
        n_det = int(det[:, i].sum())
        pmax = float(p_cnn[:, i].max())
        snaps = []
        for tr in tracks:
            if tr.n_upd < 5:
                continue
            s = tr.s.copy()
            s[2] -= terrain.height(s[0], s[1])                                         # height above the ground
            snaps.append(TrackSnapshot(tr.id, s, tr.P.copy(), tr.n_upd, n_det, pmax, detected_s, duty))
        dets = [] if snaps or not heard else [Detection(int(np.argmax(p_cnn[:, i])), pmax, detected_s, duty)]
        core.tick(float(t), snaps, dets)
        if border is not None:
            border.update(float(t), snaps)
        frames.append({"t": round(float(t), 2),
                       "tracks": [{"id": tr.id, "x": round(float(tr.s[0]), 1), "y": round(float(tr.s[1]), 1),
                                   "z": round(float(tr.s[2]), 1), "vx": round(float(tr.s[3]), 2),
                                   "vy": round(float(tr.s[4]), 2), "vz": round(float(tr.s[5]), 2),
                                   "sx": round(float(np.sqrt(tr.P[0, 0])), 1), "sy": round(float(np.sqrt(tr.P[1, 1])), 1),
                                   "sz": round(float(np.sqrt(tr.P[2, 2])), 1), "n": tr.n_upd}
                                  for tr in tracks if tr.n_upd >= 5]})
    return frames, core.events


def package(scn, shared, clusters, truth, tf, p_cnn, det, use, aud, U, coh, frames, events):
    terrain, centres = shared["terrain"], shared["centres"]
    K = len(centres)
    drones = []
    for i, (q, t, p, _shaft) in enumerate(shared["tracks"]):
        m = (t >= 0) & (t <= shared["duration"])
        step = max(1, round(len(t[m]) / (len(tf) + 1)))
        tt, pp = t[m][::step], p[m][::step]
        drones.append({"cls": scn["drones"][i]["cls"], "level_db": round(float(q["level"]), 1),
                       "blades": int(q["blades"]), "rotors": int(q.get("rotors", 1)),
                       "t": np.round(tt, 2).tolist(), "p": np.round(pp, 1).tolist(),
                       "agl": np.round(pp[:, 2] - terrain.height(pp[:, 0], pp[:, 1]), 1).tolist()})
    screen = [[{"blocked": c[4][i]["blocked"].tolist(),
                "g": np.round(np.clip(c[4][i]["g"], -50, 50), 2).tolist()} for i in range(len(shared["tracks"]))]
              for c in clusters]
    # direction error of used directions against the first drone (emission-time truth)
    err = np.full((K, len(tf)), np.nan)
    if "d0/c0/u" in truth:
        for k in range(K):
            e = angle_deg(U[k], truth[f"d0/c{k}/u"])
            err[k] = np.where(use[k], e, np.nan)
    track_err = []
    if drones:
        pe = truth["d0/pe"]
        for i, f in enumerate(frames):
            if f["tracks"]:
                tr = min(f["tracks"], key=lambda r: np.hypot(r["x"] - pe[i, 0], r["y"] - pe[i, 1]))
                track_err.append(float(np.hypot(tr["x"] - pe[i, 0], tr["y"] - pe[i, 1])))

    def first(a):
        return round(float(tf[np.argmax(a)]), 1) if np.any(a) else None

    sites = scn.get("sites", [])
    entered = None
    if drones and sites:
        p = np.array(drones[0]["p"])
        tt = np.array(drones[0]["t"])
        inside = np.zeros(len(tt), bool)
        for s in sites:
            inside |= np.hypot(p[:, 0] - s["x"], p[:, 1] - s["y"]) <= s["radius_m"]
        entered = round(float(tt[np.argmax(inside)]), 1) if inside.any() else None
    alerts = [{"t": e["t"], "tier": e["tier"], "to": e["to"], "text": e["text"]} for e in events if e["tier"] != "log"]
    take_cover = next((a["t"] for a in alerts if a["tier"] == "take_cover"), None)
    summary = {
        "first_audible_s": first(aud.any(0)), "first_cnn_s": first(det.any(0)),
        "first_track_s": next((f["t"] for f in frames if f["tracks"]), None),
        "first_alert_s": alerts[0]["t"] if alerts else None, "first_take_cover_s": take_cover,
        "drone_enters_site_s": entered,
        "warning_s": round(entered - take_cover, 1) if entered is not None and take_cover is not None else None,
        "median_direction_error_deg": round(float(np.nanmedian(err)), 1) if np.isfinite(err).any() else None,
        "median_track_error_m": round(float(np.median(track_err)), 1) if track_err else None,
        "cnn_frames_vs_audible": {"audible": int(aud.sum()), "cnn_drone": int(det.sum()),
                                  "cnn_on_audible": int((det & aud).sum()), "cnn_on_inaudible": int((det & ~aud).sum())},
        "per_cluster": [{"k": k, "ground_m": round(float(terrain.height(*centres[k, :2])), 1),
                         "audible_frac": round(float(aud[k].mean()), 3), "cnn_frac": round(float(det[k].mean()), 3),
                         "blocked_frac": round(float(np.mean(screen[k][0]["blocked"])) if screen[k] else 0.0, 3),
                         "first_cnn_s": first(det[k])} for k in range(K)],
    }
    return {"scenario": scn, "duration": round(shared["duration"], 2), "frame_hz": FRAME_HZ,
            "t": np.round(tf, 2).tolist(),
            "clusters": [{"x": round(float(c[0]), 1), "y": round(float(c[1]), 1), "z": round(float(c[2]), 2)} for c in centres],
            "drones": drones, "screen": screen,
            "p_cnn": np.round(p_cnn, 3).tolist(), "used": use.astype(int).tolist(), "audible": aud.astype(int).tolist(),
            "u": np.round(np.where(use[..., None], U, 0), 3).tolist(), "coh": np.round(coh, 2).tolist(),
            "frames": frames, "alerts": alerts, "summary": summary}
