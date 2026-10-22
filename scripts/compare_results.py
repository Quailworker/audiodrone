"""Compare a fresh evaluation with the reference results in results/<run>/eval_<split>.json.

    python scripts/compare_results.py models/eval results

Tolerances allow for GPU non-determinism, bf16 vs fp32 (CPU) and 1-LSB render differences across platforms.
Exits 1 if any number is outside its tolerance.
"""
import json
import sys
from pathlib import Path

RUNS = ["B_logf", "A_mel", "C_small"]
SPLITS = ["test_sim", "test_heldout"]
TOL_F1, TOL_PDET = 0.01, 0.03


def main():
    new_root, ref_root = Path(sys.argv[1]), Path(sys.argv[2])
    bad = 0
    print(f"{'run':8} {'split':12} {'class':15} {'F1 new':>7} {'F1 ref':>7}  {'FA/h new':>8} {'FA/h ref':>8}")
    for run in RUNS:
        for split in SPLITS:
            new_file, ref_file = new_root / run / f"eval_{split}.json", ref_root / run / f"eval_{split}.json"
            if not new_file.exists():
                print(f"{run} {split}: missing {new_file}")
                bad += 1
                continue
            new, ref = json.loads(new_file.read_text()), json.loads(ref_file.read_text())
            if new["scenes"] != ref["scenes"]:
                print(f"{run} {split}: {new['scenes']} scenes, reference has {ref['scenes']}")
                bad += 1
            for c, r in ref["frame"].items():
                n = new["frame"][c]
                fa_n = new["false_alarms_per_hour_no_drone_scenes"][c]
                fa_r = ref["false_alarms_per_hour_no_drone_scenes"][c]
                flag = ""
                if abs(n["f1"] - r["f1"]) > TOL_F1:
                    flag += " F1!"
                if abs(fa_n - fa_r) > max(3.0, 0.15 * fa_r):
                    flag += " FA!"
                print(f"{run:8} {split:12} {c:15} {n['f1']:7.3f} {r['f1']:7.3f}  {fa_n:8.1f} {fa_r:8.1f}{flag}")
                bad += bool(flag)
            for c, bins in ref["p_detect_vs_range_m"].items():
                for b, r in bins.items():
                    n = new["p_detect_vs_range_m"][c][b]
                    if r is not None and (n is None or abs(n - r) > TOL_PDET):
                        print(f"{run:8} {split:12} p_detect {c} {b} m: {n} vs {r}")
                        bad += 1
    print("MATCH" if not bad else f"{bad} differences outside tolerance")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
