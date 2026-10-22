"""Simulated scenes -> random 10 s crops with frame targets and an ignore mask (design.md 5.8.6)."""
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal
from torch.utils.data import Dataset

from .model import CLASSES, HOP, SR

# A frame is positive when at least 2 separate tone regions stand >= POS_DB above everything else,
# negative when no region stands >= NEG_DB above it, and ignored in between (sim/scene.py frame_labels).
POS_DB, NEG_DB = 6.0, 0.0


def scene_list(roots, split):
    items = []
    for root in roots:
        for js in sorted((Path(root) / split).glob("*.json")):
            if js.with_suffix(".flac").exists() and js.with_suffix(".npz").exists():
                items.append(js.with_suffix(""))
    return items


def frame_targets(npz, n_frames):
    """(n_frames, C) targets and mask from the per-category tone-line SNRs."""
    y = np.zeros((n_frames, len(CLASSES)), np.float32)
    mask = np.ones_like(y)
    for c, cat in enumerate(CLASSES):
        key = f"snr_lines/{cat}"
        if key not in npz:
            continue
        lines = npz[key].astype(np.float32)[:n_frames]
        lines = np.pad(lines, ((0, n_frames - len(lines)), (0, 0)), constant_values=-100)
        y[:, c] = lines[:, 1] >= POS_DB
        mask[:, c] = (lines[:, 1] >= POS_DB) | (lines[:, 0] < NEG_DB)
    return y, mask


class SceneCrops(Dataset):
    def __init__(self, roots, split, crop_s=10.0, train=True, crops_per_scene=4, seed=0):
        self.items = scene_list(roots, split)
        self.crop = int(crop_s * SR)
        self.train = train
        self.n = len(self.items) * (crops_per_scene if train else 1)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def _augment(self, x, rng):
        x = x * 10 ** (rng.uniform(-25, 15) / 20) * rng.choice([-1, 1])
        if rng.uniform() < 0.7:                  # unknown mic response: random shelf/tilt EQ
            fc = rng.uniform(80, 3000)
            b, a = signal.butter(1, fc / (SR / 2), btype=rng.choice(["low", "high"]))
            mix = rng.uniform(0.0, 0.7)
            x = (1 - mix) * x + mix * signal.lfilter(b, a, x)
        return x.astype(np.float32)

    def __getitem__(self, i):
        stem = self.items[i % len(self.items)]
        rng = np.random.default_rng(None if self.train else i)
        info = sf.info(str(stem) + ".flac")
        npz = np.load(str(stem) + ".npz")
        n_total = info.frames
        if n_total <= self.crop:
            start = 0
        elif self.train:
            # bias crops towards frames where something labelled is audible
            pos = np.zeros(len(npz["frame_t"]), bool)
            for cat in CLASSES:
                if f"snr_lines/{cat}" in npz:
                    pos |= npz[f"snr_lines/{cat}"][:, 1].astype(np.float32) >= POS_DB
            if pos.any() and rng.uniform() < 0.6:
                centre = int(rng.choice(np.flatnonzero(pos))) * HOP
                start = int(np.clip(centre - rng.integers(0, self.crop), 0, n_total - self.crop))
            else:
                start = int(rng.integers(0, n_total - self.crop))
        else:
            start = (n_total - self.crop) // 2
        x, _ = sf.read(str(stem) + ".flac", start=start, frames=self.crop, dtype="float32")
        x = np.pad(x, (0, self.crop - len(x)))
        if self.train:
            x = self._augment(x, rng)
        n_frames = self.crop // HOP + 1
        f0 = start // HOP
        y_full, m_full = frame_targets(npz, f0 + n_frames)
        return torch.from_numpy(x), torch.from_numpy(y_full[f0:f0 + n_frames]), torch.from_numpy(m_full[f0:f0 + n_frames])


def load_meta(stem):
    return json.loads(Path(str(stem) + ".json").read_text())
