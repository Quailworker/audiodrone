"""A full chain result (terra/pipeline.run) -> the compact replay the Border watch screen plays.

Chain output at REPLAY_HZ (tracks, which clusters' CNN fired, the bearings used), the alerts, and the simulated
truth kept apart (the screen shows it only with SIM TRUTH on). Used for the built-in incidents
(scripts/border_incidents.py) and for scenarios built on the screen (terra/server.py, build mode).
"""
import numpy as np

REPLAY_HZ = 5


def compact(res, scn, meta):
    """meta: id, name, t0, sector, neighbour, and optionally cluster_index (positions in the border deployment)."""
    step = int(res["frame_hz"] // REPLAY_HZ)
    K = len(res["clusters"])
    frames = []
    for i in range(0, len(res["t"]), step):
        f = res["frames"][i]
        bearings = [[k, *res["u"][k][i]] for k in range(K) if res["used"][k][i]]
        frames.append({"t": f["t"],
                       "tracks": [{k: tr[k] for k in ("id", "x", "y", "z", "vx", "vy", "vz", "sx", "sy", "sz", "n")}
                                  for tr in f["tracks"]],
                       "cnn": "".join("1" if res["p_cnn"][k][i] >= 0.5 else "0" for k in range(K)),
                       "bearings": bearings})
    truth = []
    for d in res["drones"]:
        tt = np.asarray(d["t"])
        keep = np.searchsorted(tt, np.arange(0, res["duration"], 1 / REPLAY_HZ))
        keep = np.unique(np.clip(keep, 0, len(tt) - 1))
        truth.append({"cls": d["cls"], "t": tt[keep].round(2).tolist(), "p": np.asarray(d["p"])[keep].round(1).tolist(),
                      "agl": np.asarray(d["agl"])[keep].round(1).tolist()})
    s = res["summary"]
    return {"id": meta["id"], "name": meta["name"], "sector": meta["sector"], "neighbour": meta.get("neighbour", ""),
            "lat": scn["origin"][0], "lon": scn["origin"][1], "half_m": scn["half_m"], "t0": meta.get("t0", 0),
            "duration": res["duration"], "replay_hz": REPLAY_HZ, "chain": res.get("chain"),
            "clusters": res["clusters"], "cluster_index": [int(i) for i in meta.get("cluster_index", [])],
            "border": scn.get("border", []), "sites": scn.get("sites", []), "frames": frames, "alerts": res["alerts"],
            "truth": truth, "ambient": scn.get("ambient"), "built": bool(meta.get("built")),
            "summary": {k: s[k] for k in ("first_audible_s", "first_cnn_s", "first_track_s", "first_alert_s",
                                          "median_direction_error_deg", "median_track_error_m")}}
