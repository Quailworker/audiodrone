"""Live detector for the field test: microphone (or a recording played back in real time) -> CNN ->
frequency tracker -> direction (with a 4-mic cluster). Records the raw audio as it goes.

    python -m live.run --list                                     # capture devices
    python -m live.run --device hw:1,0 --channels 1 --record field/flyover1.wav
    python -m live.run --device hw:1,0 --channels 4 --cluster flat --record field/c0.wav
    python -m live.run --replay results/real_world/clip.wav       # test without a mic, real-time speed
    python -m live.run --replay clip.wav --speed 0                # as fast as possible (offline check)

Runs on the GPU when there is one (0.5 s updates, 4 s CNN window), else the CPU (1 s updates, 2 s window).
If the analysis falls behind real time it skips updates (logged as "skipped") but never stops recording.

Every --update-s (0.5 s) it prints one line and appends one JSON record to --log:
  p_drone    CNN drone probability (max of the two drone heads) over the last update, model 1 by default
  tone       the strongest harmonic family the frequency tracker holds now (f0, Hz) and its comb score
  pitch      the tone's trend over the last 2 s in %/s: falling while the drone passes is the Doppler
             fly-by; on one mic it is only a supporting cue (an RPM change looks the same, design 12.3)
  bearing    with --channels 4: azimuth / elevation from the reflection-aware beamformer, steered on the
             tracker's harmonic lines (loc/run_doa.py bf_trk), and its coherence

Audio input: sounddevice (PortAudio) when installed, else `arecord` (ALSA, Linux) as a subprocess. The
level reaching the CNN matters (it was trained on calibrated levels): --gain-db scales the input.
Nothing here decides alerts; this is the per-cluster front of the chain (ARCHITECTURE.md).
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal

from cnn.model import CLASSES, SR, DroneCNN14
from loc.freqtrack import HarmonicTracker

FS_LOC = 8000                     # tracker and direction finding rate (as the localisation scenes)
TRK_WIN_S = 8.0                   # tracker window ending now
CNN_WIN_S = {"cuda": 4.0, "cpu": 2.0}        # CNN window; the CPU one keeps a laptop near real time
UPDATE_S = {"cuda": 0.5, "cpu": 1.0}         # measured per update: GPU 0.19 s, CPU ~1.3 s (busy desktop)
DRONE = [CLASSES.index("drone_electric"), CLASSES.index("drone_engine")]


# --- audio sources -----------------------------------------------------------------------------

class Replay:
    """A WAV file delivered in blocks at real-time speed (speed 0: as fast as possible)."""

    def __init__(self, path, block_s, speed=1.0):
        self.f = sf.SoundFile(path)
        self.rate, self.channels = self.f.samplerate, self.f.channels
        self.n = int(block_s * self.rate)
        self.speed, self.t0, self.done = speed, time.time(), 0

    def read(self):
        x = self.f.read(self.n, dtype="float32", always_2d=True)
        if len(x) == 0:
            return None
        self.done += len(x)
        if self.speed:
            wait = self.t0 + self.done / self.rate / self.speed - time.time()
            if wait > 0:
                time.sleep(wait)
        return x

    def close(self):
        self.f.close()


class Arecord:
    """ALSA capture through `arecord` (no PortAudio needed): 32-bit little-endian interleaved."""

    def __init__(self, device, rate, channels, block_s):
        if not shutil.which("arecord"):
            raise RuntimeError("arecord not found: install alsa-utils, or sounddevice + PortAudio")
        self.rate, self.channels = rate, channels
        self.n = int(block_s * rate)
        self.p = subprocess.Popen(["arecord", "-q", "-D", device, "-f", "S32_LE", "-c", str(channels),
                                   "-r", str(rate), "-t", "raw"], stdout=subprocess.PIPE)

    def read(self):
        need = self.n * self.channels * 4
        buf = self.p.stdout.read(need)
        if not buf:
            return None
        x = np.frombuffer(buf[: len(buf) // (4 * self.channels) * 4 * self.channels], "<i4")
        return (x.reshape(-1, self.channels) / 2 ** 31).astype(np.float32)

    def close(self):
        self.p.terminate()


class SoundDevice:
    def __init__(self, device, rate, channels, block_s):
        import sounddevice as sd
        self.rate, self.channels = rate, channels
        self.n = int(block_s * rate)
        self.s = sd.InputStream(device=device, samplerate=rate, channels=channels, dtype="float32")
        self.s.start()

    def read(self):
        x, _ = self.s.read(self.n)
        return x

    def close(self):
        self.s.stop()
        self.s.close()


def open_source(a, block_s):
    if a.replay:
        return Replay(a.replay, block_s, a.speed)
    try:
        return SoundDevice(a.device, a.rate, a.channels, block_s)
    except (ImportError, OSError):                    # no PortAudio on this machine
        return Arecord(a.device or "default", a.rate, a.channels, block_s)


def list_devices():
    try:
        import sounddevice as sd
        print(sd.query_devices())
    except (ImportError, OSError):
        print(subprocess.run(["arecord", "-l"], capture_output=True, text=True, check=False).stdout or "no arecord")


# --- processing ------------------------------------------------------------------------------------

class Resampler:
    """Streaming polyphase resampling of one channel (scipy resample_poly on overlapping blocks).

    Each call resamples the pending input plus MARGIN samples of context on both sides and emits only the
    output whose filter support lies inside it, so block edges leave no clicks. Input positions are kept
    at multiples of `down`, where input and output samples line up exactly. Output lags input by MARGIN."""

    MARGIN_S = 0.05

    def __init__(self, fs_in, fs_out):
        g = np.gcd(int(fs_in), int(fs_out))
        self.up, self.down = int(fs_out) // g, int(fs_in) // g
        self.m = max(1, round(self.MARGIN_S * fs_in / self.down)) * self.down
        self.buf = np.zeros(0, np.float32)
        self.first = True

    def __call__(self, x):
        if self.up == self.down == 1:
            return x
        self.buf = np.concatenate([self.buf, x.astype(np.float32)])
        n = len(self.buf)
        b = (n - self.m) // self.down * self.down                # last input position with full right context
        a = 0 if self.first else self.m
        if b <= a:
            return np.zeros(0, np.float32)
        y = signal.resample_poly(self.buf[: b + self.m], self.up, self.down)
        out = y[a * self.up // self.down: b * self.up // self.down]
        self.buf = self.buf[b - self.m:]
        self.first = False
        return out.astype(np.float32)


class Live:
    def __init__(self, a, rate, channels):
        self.a, self.rate, self.channels = a, rate, channels
        self.dev = a.device_torch
        self.model = DroneCNN14(a.frontend).to(self.dev).eval()
        self.model.load_state_dict(torch.load(a.model, map_location=self.dev))
        self.r32 = Resampler(rate, SR)
        self.r8 = [Resampler(rate, FS_LOC) for _ in range(channels)]
        self.x32 = deque(maxlen=int(CNN_WIN_S[self.dev] * SR))
        self.x8 = [deque(maxlen=int(TRK_WIN_S * FS_LOC)) for _ in range(channels)]
        self.trk = HarmonicTracker(FS_LOC)
        self.tones = deque(maxlen=int(2.0 / a.update_s) + 1)            # (t, f0) for the pitch trend
        self.bf = None
        if channels == 4 and a.cluster:
            from loc.doa import BeamformerDOA
            from sim.layouts import MOUNT_Z, cluster_offsets
            self.bf = BeamformerDOA(cluster_offsets(a.cluster), FS_LOC, MOUNT_Z[a.mount], device=self.dev)

    def push(self, x):
        g = 10 ** (self.a.gain_db / 20)
        self.x32.extend(self.r32(x[:, self.a.ref] * g))
        for c in range(self.channels):
            self.x8[c].extend(self.r8[c](x[:, c] * g))

    @torch.no_grad()
    def cnn(self):
        w = torch.from_numpy(np.asarray(self.x32, np.float32))[None].to(self.dev)
        p = torch.sigmoid(self.model(w).float())[0].cpu().numpy()                   # (frames at 100/s, classes)
        recent = p[-int(self.a.update_s * 100):]
        return float(recent[:, DRONE].max(1).mean()), float(recent[:, CLASSES.index("ground_engine")].mean())

    def track(self, t_now):
        x = np.asarray(self.x8[self.a.ref], np.float32)
        span = len(x) / FS_LOC
        tc = np.arange(0.25, span - 0.2, 0.1)                       # frames whose 0.4 s window is complete
        if len(tc) < 5:
            return None
        r = self.trk.run(x, tc, n_tones=1)
        f = r["f0_fine"][-1, 0] if np.isfinite(r["f0_fine"][-1, 0]) else r["f0"][-1, 0]
        if not np.isfinite(f):
            return None
        return {"f0": round(float(f), 2), "score": round(float(r["score"][-1, 0]), 1), "t_frame": t_now - (span - tc[-1])}

    def bearing(self, f0):
        from loc.run_doa import tracker_weights
        x = np.stack([np.asarray(c, np.float32) for c in self.x8])
        tc = np.array([len(x[0]) / FS_LOC - 0.25])
        fr = {"c0/f0": np.array([[f0]]), "c0/f0_fine": np.array([[f0]])}
        r = self.bf.run(x, tc, chunk=1, weights=tracker_weights(fr, 0, self.bf.base.f))
        u = r["u"][0]
        return {"az": round(float(np.degrees(np.arctan2(u[1], u[0]))) % 360, 1),
                "el": round(float(np.degrees(np.arcsin(np.clip(u[2], -1, 1)))), 1), "coh": round(float(r["coh"][0]), 2)}

    def update(self, t_now):
        rec = {"t": round(t_now, 2)}
        if len(self.x32) >= SR:                                     # at least 1 s for the CNN
            rec["p_drone"], rec["p_ground"] = (round(v, 3) for v in self.cnn())
        tone = self.track(t_now)
        if tone:
            rec["tone"] = tone
            self.tones.append((t_now, tone["f0"]))
            ts = [(t, f) for t, f in self.tones if abs(np.log(f / tone["f0"])) < 0.05]   # same family
            if len(ts) >= 3:
                tt, ff = np.array(ts).T
                rec["pitch_pct_s"] = round(float(np.polyfit(tt - tt[-1], np.log(ff), 1)[0] * 100), 2)
            if self.bf is not None:
                rec["bearing"] = self.bearing(tone["f0"])
        return rec


def line(rec, threshold):
    p = rec.get("p_drone")
    bar = "" if p is None else ("#" * round(p * 20)).ljust(20)
    flag = "DRONE" if p is not None and p >= threshold else "     "
    s = f"{rec['t']:8.1f}s  {flag} [{bar}] {'' if p is None else f'{p:.2f}'}"
    if "tone" in rec:
        s += f"  tone {rec['tone']['f0']:7.1f} Hz (score {rec['tone']['score']:4.1f})"
    if "pitch_pct_s" in rec:
        s += f"  pitch {rec['pitch_pct_s']:+.2f} %/s"
    if "bearing" in rec:
        b = rec["bearing"]
        s += f"  az {b['az']:5.1f} el {b['el']:4.1f} coh {b['coh']:.2f}"
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true", help="list capture devices and exit")
    ap.add_argument("--replay", help="WAV to play back instead of a microphone")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed (0 = as fast as possible)")
    ap.add_argument("--device", default=None, help="capture device (sounddevice index/name or ALSA hw:X,Y)")
    ap.add_argument("--rate", type=int, default=48000)
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--ref", type=int, default=0, help="channel used for the CNN and the tracker")
    ap.add_argument("--cluster", choices=["flat", "tetra"], help="with --channels 4: the cluster's geometry")
    ap.add_argument("--mount", default="ground", help="cluster mount (sim/layouts.py MOUNT_Z)")
    ap.add_argument("--model", default="models/B_logf_best.pt")
    ap.add_argument("--frontend", default="logf")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--gain-db", type=float, default=0.0)
    ap.add_argument("--update-s", type=float, default=None, help="seconds per update (default 0.5 GPU, 1.0 CPU)")
    ap.add_argument("--record", help="write the raw input to this WAV (24-bit)")
    ap.add_argument("--log", default=None, help="JSONL log (default: next to --record, else live_log.jsonl)")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this long (0 = until Ctrl-C / end)")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        list_devices()
        return
    a.device_torch = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    a.update_s = a.update_s or UPDATE_S[a.device_torch]
    src = open_source(a, a.update_s)
    live = Live(a, src.rate, src.channels)
    log_path = Path(a.log or (Path(a.record).with_suffix(".jsonl") if a.record else "live_log.jsonl"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rec_f = None
    if a.record:
        Path(a.record).parent.mkdir(parents=True, exist_ok=True)
        rec_f = sf.SoundFile(a.record, "w", src.rate, src.channels, "PCM_24")
    print(f"input {src.rate} Hz x {src.channels} ch, model {a.model} on {a.device_torch}; Ctrl-C to stop", file=sys.stderr)
    t_audio, t_wall0, skipped = 0.0, time.time(), 0
    try:
        with open(log_path, "a") as log:
            while True:
                x = src.read()
                if x is None:
                    break
                if rec_f is not None:
                    rec_f.write(x)
                live.push(x)
                t_audio += len(x) / src.rate
                realtime = not a.replay or a.speed == 1.0
                if realtime and time.time() - t_wall0 - t_audio > a.update_s:
                    skipped += 1                    # behind real time: keep recording, skip this analysis
                    log.write(json.dumps({"t": round(t_audio, 2), "skipped": True}) + "\n")
                    continue
                rec = live.update(t_audio)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(line(rec, a.threshold), flush=True)
                if a.seconds and t_audio >= a.seconds:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        if rec_f is not None:
            rec_f.close()
    print(f"log: {log_path}" + (f", audio: {a.record}" if a.record else "")
          + (f"; {skipped} updates skipped to keep up with real time" if skipped else ""), file=sys.stderr)


if __name__ == "__main__":
    main()
