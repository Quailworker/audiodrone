"""Fine-tune CNN14 on simulated scenes (design.md 5.7, 5.8).

    python -m cnn.train --data data/sim_v1 --frontend logf --out runs/logf_v1

Stage 1 trains bn0 + fc1 + the new head with the body frozen; stage 2 unfreezes everything at a low LR.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import SceneCrops
from .model import CLASSES, DroneCNN14


def masked_bce(logits, y, mask, pos_weight):
    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pos_weight)
    return (loss * mask).sum() / mask.sum().clamp(min=1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp = np.zeros(len(CLASSES)); fp = np.zeros(len(CLASSES)); fn = np.zeros(len(CLASSES))
    losses = []
    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        logits = logits.float()[:, : y.shape[1]]
        losses.append(masked_bce(logits, y, m, None).item())
        pred = (logits > 0).float()
        tp += ((pred * y) * m).sum((0, 1)).cpu().numpy()
        fp += ((pred * (1 - y)) * m).sum((0, 1)).cpu().numpy()
        fn += (((1 - pred) * y) * m).sum((0, 1)).cpu().numpy()
    model.train()
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return {"loss": float(np.mean(losses)), **{f"f1/{c}": float(v) for c, v in zip(CLASSES, f1)},
            **{f"prec/{c}": float(t / max(t + p, 1)) for c, t, p in zip(CLASSES, tp, fp)},
            **{f"rec/{c}": float(t / max(t + n, 1)) for c, t, n in zip(CLASSES, tp, fn)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--frontend", default="logf", choices=["logf", "mel"])
    ap.add_argument("--checkpoint", default="weights/Cnn14_DecisionLevelMax_mAP=0.385.pth")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--stage1_steps", type=int, default=1500)
    ap.add_argument("--stage2_steps", type=int, default=15000)
    ap.add_argument("--lr1", type=float, default=1e-3)
    ap.add_argument("--lr2", type=float, default=1e-4)
    ap.add_argument("--eval_every", type=int, default=1000)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    model = DroneCNN14(a.frontend, a.checkpoint).to(device)

    train = SceneCrops(a.data, "train", train=True)
    val = SceneCrops(a.data, "val", train=False)
    print(f"train scenes {len(train.items)}  val scenes {len(val.items)}", flush=True)
    tl = DataLoader(train, a.batch, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=True)
    vl = DataLoader(val, a.batch, num_workers=a.workers)
    pos_weight = torch.tensor([2.0, 3.0, 3.0], device=device)

    for p in model.body_parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(model.head_parameters(), lr=a.lr1, weight_decay=1e-4)
    total = a.stage1_steps + a.stage2_steps
    step, best, t0, run = 0, -1.0, time.time(), []
    log = open(out / "log.jsonl", "a")
    while step < total:
        for x, y, m in tl:
            if step == a.stage1_steps:          # stage 2: unfreeze, cosine decay
                for p in model.parameters():
                    p.requires_grad = True
                opt = torch.optim.AdamW(model.parameters(), lr=a.lr2, weight_decay=1e-4)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.stage2_steps)
            x, y, m = x.to(device, non_blocking=True), y.to(device), m.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss = masked_bce(logits.float()[:, : y.shape[1]], y, m, pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if step >= a.stage1_steps:
                sched.step()
            run.append(loss.item())
            step += 1
            if step % 100 == 0:
                print(f"step {step}/{total} loss {np.mean(run):.4f} {(time.time() - t0) / step:.2f}s/step", flush=True)
                run = []
            if step % a.eval_every == 0 or step == total:
                ev = evaluate(model, vl, device)
                score = float(np.mean([ev[f"f1/{c}"] for c in CLASSES]))
                ev.update(step=step, mean_f1=score)
                log.write(json.dumps(ev) + "\n"); log.flush()
                print("EVAL", json.dumps(ev), flush=True)
                torch.save(model.state_dict(), out / "last.pt")
                if score > best:
                    best = score
                    torch.save(model.state_dict(), out / "best.pt")
            if step >= total:
                break


if __name__ == "__main__":
    main()
