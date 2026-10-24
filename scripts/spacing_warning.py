"""Real warning time per head-on spacing run (results/spacing/README.md).

    python scripts/spacing_warning.py [results/spacing]

A take-cover only counts once it is not a false "over it now" (drone more than 2 site radii out). The drone's
true distance is taken from the straight head-on flight in scripts/spacing_study.py: 3 km out at t = 0, at the
class speed.
"""
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.spacing_study import CLASSES, SITE_R

START_M = 3000.0


def operator_alerts(path):
    al = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    cut = max([i for i in range(1, len(al)) if al[i]["t"] < al[i - 1]["t"] - 1], default=0)   # older logs were appended
    return [a for a in al[cut:] if a.get("to") == "operators"]


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "results/spacing")
    print("side  cls     detect_m  first_cover  real_cover  enters  real_warning_s  false_over_now  first_advisory_m")
    for r in csv.DictReader((out / "runs_v2.csv").read_text().splitlines()):
        if r["flight"] != "head_on":
            continue
        al = operator_alerts(out / "runs" / f"v2_{r['side_m']}_{r['cls']}_head_on" / "alerts.jsonl")
        v = CLASSES[r["cls"]]["speed"]

        def dist(t, v=v):
            return max(START_M - v * t, 0.0)
        covers = [a for a in al if a["tier"] == "take_cover"]
        real = [a for a in covers if not (re.search("over it now", a["text"]) and dist(a["t"]) > 2 * SITE_R)]
        adv = [a for a in al if a["tier"] != "all_clear"]
        t_real = real[0]["t"] if real else None
        warn = f"{float(r['enters_s']) - t_real:.1f}" if t_real is not None else "-"
        print(f"{r['side_m']:>4}  {r['cls']:<6} {r['first_detect_range_m']:>8}  {covers[0]['t'] if covers else '-':>11}  "
              f"{t_real if t_real is not None else '-':>10}  {r['enters_s']:>6}  {warn:>14}  {len(covers) - len(real):>14}  "
              f"{dist(adv[0]['t']) if adv else float('nan'):>16.0f}")


if __name__ == "__main__":
    main()
