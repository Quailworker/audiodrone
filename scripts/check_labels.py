"""Count scenes per split and any rendered with an old label version.  python scripts/check_labels.py"""
import collections
import glob
import json

counts = collections.Counter()
stale = 0
for js in glob.glob("data/sim_v1/*/*.json"):
    meta = json.load(open(js))
    counts[js.replace("\\", "/").split("/")[-2]] += 1
    stale += meta.get("labels") != "snr_lines2"
print("scenes per split", dict(counts), "stale labels", stale)
