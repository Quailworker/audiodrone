"""Live front end: streaming resampler and a replayed recording through CNN + frequency tracker."""
import json

import numpy as np
import soundfile as sf
from scipy import signal

from live.run import Resampler, main


def test_streaming_resampler_matches_offline():
    fs = 44100
    x = np.random.default_rng(0).standard_normal(3 * fs).astype(np.float32)
    r = Resampler(fs, 32000)
    y = np.concatenate([r(x[i: i + 22050]) for i in range(0, len(x), 22050)])
    ref = signal.resample_poly(x, 320, 441)
    assert len(y) > 0.95 * len(ref)
    assert np.abs(y - ref[: len(y)])[2000:].max() < 1e-5


def test_replay_finds_tone(tmp_path):
    """A 6 s harmonic tone (150 Hz family) in noise, replayed as fast as possible on the CPU."""
    fs = 48000
    t = np.arange(6 * fs) / fs
    x = sum(0.6 ** k * np.sin(2 * np.pi * 150 * k * t) for k in range(1, 12)) * 0.02
    x = x + 0.002 * np.random.default_rng(1).standard_normal(len(t))
    wav, log = tmp_path / "tone.wav", tmp_path / "log.jsonl"
    sf.write(wav, x.astype(np.float32), fs)
    main(["--replay", str(wav), "--speed", "0", "--cpu", "--log", str(log), "--record", str(tmp_path / "rec.wav")])
    recs = [json.loads(s) for s in log.read_text().splitlines()]
    tones = [r["tone"]["f0"] for r in recs if "tone" in r and r["t"] > 2]
    assert len(tones) >= 3 and all(abs(f / 150 - 1) < 0.005 for f in tones), tones
    assert all("p_drone" in r for r in recs if r["t"] >= 1.5)              # CNN needs 1 s (+ resampler lag)
    assert abs(sf.info(tmp_path / "rec.wav").duration - 6.0) < 0.01
