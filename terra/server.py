"""UAE terrain tool: pick an area, place mic clusters, protected sites and drone flights in 3D, run the
whole chain on the rendered sound and play it back. Also serves the Border watch screen (/border).

    .venv/bin/python -m terra.server [--port 8790]        then open http://127.0.0.1:8790 (or /border)

Runs are kept under runs_terra/<id>/ (scenario.json, result.json, alerts.jsonl, cluster<k>.wav).
One run at a time; the rendering uses several processes, the CNN and beamformer the GPU.
"""
import argparse
import base64
import io
import json
import math
import re
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import uvicorn
from PIL import Image
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .dem import Terrain, overview
from .scenario import AMBIENT, DRONE_CLASSES

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
EXAMPLES = Path(__file__).resolve().parent / "examples"
RUNS = ROOT / "runs_terra"
BORDER_DATA = Path(__file__).resolve().parent / "data"
PRESETS = [
    {"name": "Jebel Hafeet, Al Ain", "lat": 24.06, "lon": 55.78, "half_m": 4000},
    {"name": "Jebel Jais, Ras Al Khaimah", "lat": 25.95, "lon": 56.13, "half_m": 4000},
    {"name": "Hatta, Hajar foothills", "lat": 24.80, "lon": 56.12, "half_m": 3000},
    {"name": "Dubai Marina", "lat": 25.08, "lon": 55.14, "half_m": 2000},
    {"name": "Abu Dhabi, Corniche", "lat": 24.47, "lon": 54.34, "half_m": 2000},
    {"name": "Liwa dunes", "lat": 23.10, "lon": 53.75, "half_m": 3000},
    {"name": "Fujairah coast", "lat": 25.13, "lon": 56.33, "half_m": 3000},
]
JOBS = {}
LOCK = threading.Lock()


def hillshade_png(H, cell_m, sea_level=0.5):
    """Hypsometric tint with hillshade; sea (<= sea_level m) in blue. -> PNG bytes."""
    gy, gx = np.gradient(H, cell_m)
    az, alt = np.radians(315), np.radians(45)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    shade = np.clip(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect), 0, 1)
    stops = np.array([0, 50, 150, 400, 800, 1500, 2000], float)
    cols = np.array([[222, 205, 160], [214, 190, 140], [200, 170, 120], [176, 140, 100],
                     [150, 115, 85], [120, 100, 90], [235, 235, 235]], float)
    rgb = np.stack([np.interp(H, stops, cols[:, i]) for i in range(3)], -1)
    rgb = rgb * (0.55 + 0.6 * shade[..., None])
    sea = H <= sea_level
    rgb[sea] = [86, 132, 168]
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


_overview_png = None


async def api_overview(request):
    global _overview_png
    H, (lo0, la0, lo1, la1) = overview()
    if _overview_png is None:
        _overview_png = hillshade_png(H, 1100.0)
    return JSONResponse({"bounds": [lo0, la0, lo1, la1], "png": "data:image/png;base64," + base64.b64encode(_overview_png).decode(),
                         "presets": PRESETS, "drone_classes": DRONE_CLASSES, "ambient": list(AMBIENT)})


async def api_terrain(request):
    q = request.query_params
    lat, lon, half = float(q["lat"]), float(q["lon"]), float(q.get("half_m", 3000))
    half = min(max(half, 500.0), 6000.0)
    res = max(10.0, 2 * half / 300)                                       # <= 301 x 301 vertices for the mesh
    try:
        T = Terrain(lat, lon, half_m=half, res_m=res)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    tex = hillshade_png(T.H[::-1], res)                                   # north-up image for the texture
    return JSONResponse({"lat": lat, "lon": lon, "half_m": half, "res_m": res, "n": len(T.x),
                         "h": base64.b64encode(T.H.astype(np.float32).tobytes()).decode(),
                         "min": float(T.H.min()), "max": float(T.H.max()),
                         "texture": "data:image/png;base64," + base64.b64encode(tex).decode()})


def _run_job(job_id, scn, post=None, workers=6):
    from .pipeline import run
    job = JOBS[job_id]
    out = RUNS / job_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "scenario.json").write_text(json.dumps(scn, indent=1))

    def progress(msg):
        job["progress"].append([round(time.time() - job["t0"], 1), msg])
    try:
        res = run(scn, out, progress=progress, workers=workers)
        res["id"] = job_id
        (out / "result.json").write_text(json.dumps(res))
        if post is not None:
            post(job_id, scn, res)
        job["status"] = "done"
    except Exception as e:  # noqa: BLE001 - a failed run is reported to the tool, never kills the server
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"
        (out / "error.txt").write_text(traceback.format_exc())
    finally:
        LOCK.release()


async def api_run(request):
    scn = await request.json()
    err = validate(scn)
    if err:
        return JSONResponse({"error": err}, status_code=400)
    if not LOCK.acquire(blocking=False):
        return JSONResponse({"error": "a run is already in progress; wait for it to finish"}, status_code=409)
    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    JOBS[job_id] = {"status": "running", "progress": [], "t0": time.time()}
    threading.Thread(target=_run_job, args=(job_id, scn), daemon=True).start()
    return JSONResponse({"id": job_id})


def validate(scn):
    if not scn.get("clusters") or len(scn["clusters"]) < 2:
        return "place at least two mic clusters"
    if len(scn["clusters"]) > 12:
        return "at most 12 clusters per run"
    if not scn.get("drones"):
        return "draw at least one drone flight"
    for d in scn["drones"]:
        if d.get("cls") not in DRONE_CLASSES:
            return f"unknown drone class {d.get('cls')}"
        if len(d.get("waypoints", [])) < 2:
            return "a drone flight needs at least two waypoints"
    if scn.get("ambient", "desert") not in AMBIENT:
        return "unknown ambient sound"
    return None


async def api_job(request):
    job_id = request.path_params["job_id"]
    job = JOBS.get(job_id)
    path = RUNS / job_id / "result.json"
    if job is None and not path.exists():
        return JSONResponse({"error": "unknown run"}, status_code=404)
    if job is None or job["status"] == "done":
        res = json.loads(path.read_text())
        scn_file = RUNS / job_id / "scenario.json"
        if scn_file.exists():                                  # the saved scenario carries the area name
            res["scenario"] = {**res.get("scenario", {}), **json.loads(scn_file.read_text())}
        return JSONResponse(res)
    return JSONResponse({k: v for k, v in job.items() if k != "t0"})


async def api_runs(request):
    rows = []
    for d in sorted(RUNS.glob("*/result.json"), reverse=True)[:30]:
        scn = json.loads((d.parent / "scenario.json").read_text()) if (d.parent / "scenario.json").exists() else {}
        rows.append({"id": d.parent.name, "area": scn.get("area_name", ""), "clusters": len(scn.get("clusters", [])),
                     "terrain": scn.get("terrain", True)})
    return JSONResponse(rows)


async def api_audio(request):
    job_id, k = request.path_params["job_id"], int(request.path_params["k"])
    f = RUNS / job_id / f"cluster{k}.wav"
    if not f.exists() or ".." in job_id:
        return JSONResponse({"error": "no audio"}, status_code=404)
    return FileResponse(f, media_type="audio/wav")


async def api_examples(request):
    """Example scenarios shipped with the repo (terra/examples/*.json), ready to load and run."""
    return JSONResponse([{"id": f.stem, **json.loads(f.read_text())} for f in sorted(EXAMPLES.glob("*.json"))])


async def index(request):
    return FileResponse(STATIC / "index.html")


# ---- Border watch screen (terra/static/border.html): whole land border, incidents replayed from the chain
_theatre = None


async def border_page(request):
    return FileResponse(STATIC / "border.html")


async def api_theatre(request):
    """Heights of the whole UAE on a 0.01 degree grid (rows north-up) for the theatre map."""
    global _theatre
    if _theatre is None:
        H, (lo0, la0, lo1, la1) = overview(step_deg=0.01)
        _theatre = {"bounds": [lo0, la0, lo1, la1], "step": 0.01, "rows": H.shape[0], "cols": H.shape[1],
                    "h": base64.b64encode(np.ascontiguousarray(H, np.float32).tobytes()).decode()}
    return JSONResponse(_theatre)


async def api_border(request):
    return FileResponse(BORDER_DATA / "border.json", media_type="application/json")


async def api_incidents(request):
    rows = []
    for f in sorted((BORDER_DATA / "incidents").glob("*.json")):
        d = json.loads(f.read_text())
        tiers = [a["tier"] for a in d["alerts"]]
        rows.append({k: d[k] for k in ("id", "name", "sector", "neighbour", "lat", "lon", "half_m", "t0", "duration")}
                    | {"alerts": len(tiers), "crossed": "border_cross" in tiers})
    return JSONResponse(sorted(rows, key=lambda r: r["t0"]))


def _incident(iid):
    if not iid or "/" in iid or ".." in iid:
        return None
    for d in ("incidents", "user"):
        f = BORDER_DATA / d / f"{iid}.json"
        if f.exists():
            return json.loads(f.read_text())
    return None


async def api_sitedata(request):
    """Posts, shelters and units around an incident, read through Zetaris (terra/ask.py)."""
    from starlette.concurrency import run_in_threadpool

    from .ask import site_data
    inc = _incident(request.path_params["iid"])
    if inc is None:
        return JSONResponse({"error": "unknown incident"}, status_code=404)
    return JSONResponse(await run_in_threadpool(site_data, inc))


async def api_ask(request):
    """Operator question about an incident at replay time t: {"id", "t", "question"} -> Nemotron's answer."""
    from .ask import ask
    q = await request.json()
    inc = _incident(str(q.get("id", "")))
    question = str(q.get("question", "")).strip()[:500]
    if inc is None or not question:
        return JSONResponse({"error": "need a known incident id and a question"}, status_code=400)
    try:
        return JSONResponse(await ask(inc, float(q.get("t", 0.0)), question))
    except Exception as e:  # noqa: BLE001 - shown to the operator, never kills the server
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


async def api_federated(request):
    """A question across the separate stores, answered by one Zetaris query (terra/federated.py)."""
    from starlette.concurrency import run_in_threadpool

    from alerting.config import _load_env_file

    from .federated import QUERIES
    q, iid = request.query_params.get("q", ""), request.query_params.get("id", "")
    fn = QUERIES.get(q)
    if fn is None:
        return JSONResponse({"error": f"unknown query; one of {sorted(QUERIES)}"}, status_code=400)
    _load_env_file()
    try:
        return JSONResponse(await run_in_threadpool(fn) if q == "overview" else await run_in_threadpool(fn, iid))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:  # noqa: BLE001 - Zetaris down or a query error: shown on the screen
        msg = re.findall(r"<<<(.*?)>>>", str(e))
        return JSONResponse({"error": f"Zetaris query failed: {msg[0] if msg else type(e).__name__}"}, status_code=502)


async def api_incident(request):
    iid = request.path_params["iid"]
    if "/" in iid or ".." in iid:
        return JSONResponse({"error": "unknown incident"}, status_code=404)
    for d in ("incidents", "user"):
        f = BORDER_DATA / d / f"{iid}.json"
        if f.exists():
            return FileResponse(f, media_type="application/json")
    return JSONResponse({"error": "unknown incident"}, status_code=404)


# ---- Build mode: place clusters and drones on the tactical screen, run the whole chain, replay it there
BUILD_MAX = {"clusters": 24, "drones": 6, "half_m": 6000.0}
STAGES = [("rendering sound", "Simulating what every microphone hears"), ("rendered cluster", "Simulating what every microphone hears"),
          ("CNN detection", "Neural network listening for drones"), ("frequency tracking", "Finding each drone's tones"),
          ("direction finding", "Working out the direction at each cluster"), ("tracking and alerts", "Tracking and sending alerts"),
          ("done", "Done")]


def _build_scenario(q):
    """Client request -> a full scenario (terra/scenario.py format), with the border lines when near the border."""
    from .border import load as load_border
    from .border import local_border, nearest_border_point
    lat, lon = float(q["origin"][0]), float(q["origin"][1])
    pts = [p for p in q.get("clusters", [])] + [w for d in q.get("drones", []) for w in d.get("waypoints", [])]
    extent = max([max(abs(float(x)), abs(float(y))) for x, y in pts] or [1000.0])
    half = float(min(max(extent + 600.0, 1500.0), BUILD_MAX["half_m"]))
    doc = load_border()
    sector, blon, blat, _ = nearest_border_point(doc, lat, lon)
    from .dem import metres_per_degree
    m_lat, m_lon = metres_per_degree(lat)
    near = math.hypot((blon - lon) * m_lon, (blat - lat) * m_lat) < half + 3000
    return {"origin": [lat, lon], "half_m": half, "terrain": True, "mount": "ground",
            "clusters": [[round(float(x), 1), round(float(y), 1)] for x, y in q.get("clusters", [])],
            "sites": q.get("sites", []),
            "drones": [{"cls": d["cls"], "waypoints": [[round(float(x), 1), round(float(y), 1)] for x, y in d["waypoints"]],
                        "agl": float(d.get("agl", 60)), "speed": float(d.get("speed", 12)), "alt_mode": "agl"}
                       for d in q.get("drones", [])],
            "border": local_border(doc, lat, lon, half) if near else [],
            "ambient": q.get("ambient", "desert"), "temp_c": float(q.get("temp_c", 35)), "rh": float(q.get("rh", 30)),
            "seed": int(q.get("seed", 7)), "chain": "v2",
            "area_name": str(q.get("name") or "Built scenario")[:60],
            "_sector": sector["id"] if near else "SITE", "_neighbour": sector["neighbour"] if near else ""}


def _build_check(scn):
    if len(scn["clusters"]) < 2:
        return "Place at least 2 microphone clusters: a position needs two directions."
    if len(scn["clusters"]) > BUILD_MAX["clusters"]:
        return f"At most {BUILD_MAX['clusters']} clusters per run."
    if not scn["drones"]:
        return "Draw at least one drone path."
    if len(scn["drones"]) > BUILD_MAX["drones"]:
        return f"At most {BUILD_MAX['drones']} drones per run."
    for d in scn["drones"]:
        if d["cls"] not in DRONE_CLASSES:
            return "Unknown drone type."
        if len(d["waypoints"]) < 2:
            return "Each drone needs at least 2 points on its path."
    if scn["ambient"] not in AMBIENT:
        return "Unknown background noise."
    return None


async def api_build_ranges(request):
    from .build import ranges
    return JSONResponse(ranges())


async def api_build_estimate(request):
    from .build import estimate_s
    scn = _build_scenario(await request.json())
    return JSONResponse({"estimate_s": estimate_s(scn), "error": _build_check(scn), "half_m": scn["half_m"],
                         "near_border": bool(scn["border"])})


def _save_replay(job_id, scn, res):
    from .replay import compact
    rep = compact(res, scn, {"id": job_id, "name": scn["area_name"], "sector": scn["_sector"],
                             "neighbour": scn["_neighbour"], "t0": 0, "built": True})
    d = BORDER_DATA / "user"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{job_id}.json").write_text(json.dumps(rep, separators=(",", ":")))


async def api_build_run(request):
    scn = _build_scenario(await request.json())
    err = _build_check(scn)
    if err:
        return JSONResponse({"error": err}, status_code=400)
    if not LOCK.acquire(blocking=False):
        return JSONResponse({"error": "Another simulation is running. Wait for it to finish."}, status_code=409)
    job_id = "B" + time.strftime("%m%d-%H%M%S")
    JOBS[job_id] = {"status": "running", "progress": [], "t0": time.time()}
    threading.Thread(target=_run_job, args=(job_id, scn, _save_replay, 3), daemon=True).start()
    return JSONResponse({"id": job_id})


async def api_build_status(request):
    """Plain-language progress of a build run: stage, fraction, and the replay id when it is done."""
    job = JOBS.get(request.path_params["job_id"])
    if job is None:
        return JSONResponse({"error": "unknown run"}, status_code=404)
    last = job["progress"][-1][1] if job["progress"] else "starting"
    stage = next((label for key, label in STAGES if key in last), "Starting")
    frac = 0.02
    m = re.search(r"\((\d+)/(\d+)\)", last)
    if m:
        frac = 0.05 + 0.7 * int(m.group(1)) / int(m.group(2))
    else:
        frac = {"Neural network listening for drones": 0.78, "Finding each drone's tones": 0.84,
                "Working out the direction at each cluster": 0.9, "Tracking and sending alerts": 0.96,
                "Done": 1.0}.get(stage, frac)
    return JSONResponse({"status": job["status"], "stage": stage, "detail": last, "fraction": round(frac, 3),
                         "elapsed_s": round(time.time() - job["t0"]), "error": job.get("error"),
                         "replay": request.path_params["job_id"] if job["status"] == "done" else None})


async def api_build_list(request):
    d = BORDER_DATA / "user"
    rows = []
    for f in sorted(d.glob("*.json"), reverse=True)[:20] if d.exists() else []:
        r = json.loads(f.read_text())
        rows.append({"id": r["id"], "name": r["name"], "duration": r["duration"], "clusters": len(r["clusters"]),
                     "drones": len(r["truth"])})
    return JSONResponse(rows)


app = Starlette(routes=[
    Route("/", border_page),
    Route("/classic", index),
    Route("/api/overview", api_overview),
    Route("/api/terrain", api_terrain),
    Route("/api/run", api_run, methods=["POST"]),
    Route("/api/job/{job_id}", api_job),
    Route("/api/job/{job_id}/audio/{k:int}", api_audio),
    Route("/api/runs", api_runs),
    Route("/api/examples", api_examples),
    Route("/border", border_page),
    Route("/api/theatre", api_theatre),
    Route("/api/border", api_border),
    Route("/api/incidents", api_incidents),
    Route("/api/incident/{iid}", api_incident),
    Route("/api/sitedata/{iid}", api_sitedata),
    Route("/api/ask", api_ask, methods=["POST"]),
    Route("/api/federated", api_federated),
    Route("/api/build/ranges", api_build_ranges),
    Route("/api/build/estimate", api_build_estimate, methods=["POST"]),
    Route("/api/build/run", api_build_run, methods=["POST"]),
    Route("/api/build/status/{job_id}", api_build_status),
    Route("/api/build/list", api_build_list),
    Mount("/static", StaticFiles(directory=STATIC), name="static"),
], middleware=[Middleware(GZipMiddleware, minimum_size=2048)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    RUNS.mkdir(exist_ok=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
