"""UAE terrain tool: pick an area, place mic clusters, protected sites and drone flights in 3D, run the
whole chain on the rendered sound and play it back.

    .venv/bin/python -m terra.server [--port 8790]        then open http://127.0.0.1:8790

Runs are kept under runs_terra/<id>/ (scenario.json, result.json, alerts.jsonl, cluster<k>.wav).
One run at a time; the rendering uses several processes, the CNN and beamformer the GPU.
"""
import argparse
import base64
import io
import json
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import uvicorn
from PIL import Image
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .dem import Terrain, overview
from .scenario import AMBIENT, DRONE_CLASSES

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
EXAMPLES = Path(__file__).resolve().parent / "examples"
RUNS = ROOT / "runs_terra"
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


def _run_job(job_id, scn):
    from .pipeline import run
    job = JOBS[job_id]
    out = RUNS / job_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "scenario.json").write_text(json.dumps(scn, indent=1))

    def progress(msg):
        job["progress"].append([round(time.time() - job["t0"], 1), msg])
    try:
        res = run(scn, out, progress=progress)
        res["id"] = job_id
        (out / "result.json").write_text(json.dumps(res))
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


app = Starlette(routes=[
    Route("/", index),
    Route("/api/overview", api_overview),
    Route("/api/terrain", api_terrain),
    Route("/api/run", api_run, methods=["POST"]),
    Route("/api/job/{job_id}", api_job),
    Route("/api/job/{job_id}/audio/{k:int}", api_audio),
    Route("/api/runs", api_runs),
    Route("/api/examples", api_examples),
    Mount("/static", StaticFiles(directory=STATIC), name="static"),
])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    RUNS.mkdir(exist_ok=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
