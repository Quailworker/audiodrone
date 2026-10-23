"""Check the UAE terrain tool from the command line: run an example scenario through a running server.

    .venv/bin/python -m terra.server &                      # in another terminal, or in the background
    .venv/bin/python scripts/terra_check.py [hafeet|jais] [--url http://127.0.0.1:8790]

Prints the run's progress, its summary and its alerts, and exits non-zero if the run fails.
"""
import argparse
import json
import sys
import time
import urllib.request


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("example", nargs="?", default="hafeet")
    ap.add_argument("--url", default="http://127.0.0.1:8790")
    a = ap.parse_args()
    examples = {x["id"]: x for x in get(f"{a.url}/api/examples")}
    scn = examples[a.example]
    req = urllib.request.Request(f"{a.url}/api/run", data=json.dumps(scn).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        job = json.loads(r.read())["id"]
    print(f"run {job}: {scn.get('title', a.example)}", flush=True)
    seen = 0
    while True:
        time.sleep(3)
        d = get(f"{a.url}/api/job/{job}")
        for t, msg in d.get("progress", [])[seen:]:
            print(f"  {t:6.1f} s  {msg}", flush=True)
        seen = len(d.get("progress", []))
        if d.get("status") == "error":
            print("FAILED:", d.get("error"))
            sys.exit(1)
        if "summary" in d:
            break
    s = d["summary"]
    for k in ("first_audible_s", "first_cnn_s", "first_track_s", "first_take_cover_s", "drone_enters_site_s",
              "warning_s", "median_direction_error_deg", "median_track_error_m"):
        print(f"{k:28s} {s[k]}")
    for c in s["per_cluster"]:
        print(f"  C{c['k'] + 1}: ground {c['ground_m']} m, audible {c['audible_frac']:.0%}, CNN {c['cnn_frac']:.0%}, "
              f"blocked by terrain {c['blocked_frac']:.0%}")
    print("alerts:")
    for x in d["alerts"]:
        print(f"  {x['t']:7.2f} {x['tier']:11s} {x['text'][:110]}")
    print(f"terra check OK ({d['timing']['total_s']} s)")


if __name__ == "__main__":
    main()
