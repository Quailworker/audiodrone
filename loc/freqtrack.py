"""Harmonic frequency tracker (design.md 6, 13.1; rebuild of the lost track_freq.py).

Follows the drone's harmonic family f0, 2 f0, 3 f0, ... at one microphone, frame by frame (FRAME_HZ),
and turns the tracked pitch at several clusters into Doppler information for the Kalman filter.

1. STFT of the reference mic (8 kHz audio, Hann 0.4 s, hop 0.1 s, zero-padded to ~0.5 Hz bins).
2. Whitening: dB above a running mean over +-FLOOR_HZ, clipped to 0..CLIP_DB. Lines stand out, broadband
   noise and the overall level drop out.
3. Log-frequency grid (ratio LOG_STEP per cell; a harmonic k f0 is a fixed offset ln k from f0), sampled
   by interpolation. The window's main lobe (+-~5 Hz) is the tooth width.
4. Comb score per candidate f0 (F0_LO..F0_HI): weighted mean (HARM_DECAY^(k-1)) over k of
   S(k f0) - S((k - 1/2) f0), k = 1..K, K = min(K_MAX, f_max / f0) >= K_MIN. The gaps make it an octave check: at f0/2 half the teeth
   fall on gaps (score ~1/2); at 2 f0 the odd harmonics sit in the gaps (score < 0).
5. Viterbi over frames with a "no tone" state: pitch may drift by at most MAX_JUMP per frame, with a
   quadratic cost; entering or leaving the tone costs SWITCH. Gives a smooth path instead of jumpy peaks.
6. Notch the found family and repeat, for up to n_tones families (several drones or engines).
7. Fine pitch: parabolic peak interpolation on each harmonic, then the weighted least-squares fit
   f0 = sum(w k f_k) / sum(w k^2) (higher harmonics count more: they carry the same Hz error over k f0).

Doppler (13.1): the same emitted f0 is heard at f_k = f0 / (1 + rdot_k / c) at cluster k, so for two
clusters at the same reception time  ln f_i - ln f_j ~ -(rdot_i - rdot_j) / c.  Clusters can lock onto
different members of the family (shaft rate at one, blade rate at another), so ratios are aligned to the
nearest small rational first (`align`).
"""
import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d, uniform_filter1d

C = 343.0
WIN_S, HOP_S, NFFT = 0.4, 0.1, 16384
FMIN, FMAX = 30.0, 3900.0                  # analysed band (8 kHz audio)
FLOOR_HZ, CLIP_DB = 40.0, 30.0
LOG_STEP = 0.001                           # ln-frequency step of the grid (0.1 %)
REFINE_REL, REFINE_DB = 0.003, 8.0        # refine(): peak search +-0.3 % (>= 3 bins), at least 8 dB up
TOOTH_BINS = 1                             # +-1 bin max before sampling (grid discretisation only)
NOTCH_HZ = 6.0                             # notch half-width per harmonic (Hann 0.4 s main lobe ~ +-5 Hz)
SMOOTH_FRAMES = 3                          # power averaged over 3 frames (0.3 s) before whitening
HARM_DECAY = 0.6                           # harmonic k weighted HARM_DECAY^(k-1) (low harmonics are strongest)
GAP_WEIGHT = 0.25                          # weight of the gap (octave-check) term
# HARM_DECAY, GAP_WEIGHT, NULL_SCORE: grid on 40 tetra_pole_100 scenes (results/freqtrack/README.md);
# 0.9 / 1.0 found 40 % of audible drone frames, 0.6 / 0.25 finds 64 % at the same pitch accuracy
F0_LO, F0_HI = 40.0, 1300.0
K_MAX, K_MIN = 30, 3
MAX_JUMP, JUMP_COST, SWITCH = 0.02, 2.0, 6.0
NULL_SCORE = 5.0                           # dB: the comb score a family must beat to count as a tone
RATIOS = np.array([1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 4, 1.0, 4 / 3, 3 / 2, 2.0, 3.0, 4.0])


def spectrogram(x, fs, t_centres):
    """Power (frames, bins) in dB at the given frame centres, and the bin frequencies."""
    n = int(WIN_S * fs)
    w = signal.get_window("hann", n)
    pad = np.pad(x.astype(np.float64), (n, n))
    starts = np.clip(np.round(np.asarray(t_centres) * fs).astype(int) - n // 2 + n, 0, len(pad) - n)
    f = np.fft.rfftfreq(NFFT, 1 / fs)
    band = (f >= FMIN) & (f <= min(FMAX, 0.49 * fs))
    out = np.empty((len(starts), band.sum()), np.float32)
    for s in range(0, len(starts), 256):
        seg = np.stack([pad[i: i + n] for i in starts[s: s + 256]]) * w
        out[s: s + 256] = 10 * np.log10(np.abs(np.fft.rfft(seg, NFFT)[:, band]) ** 2 + 1e-20)
    return out, f[band]


def whiten(db, f):
    """dB above the running mean over +-FLOOR_HZ, clipped to 0..CLIP_DB (after SMOOTH_FRAMES time averaging)."""
    if SMOOTH_FRAMES > 1:
        db = 10 * np.log10(uniform_filter1d(10 ** (db / 10), SMOOTH_FRAMES, axis=0, mode="nearest"))
    width = max(3, round(2 * FLOOR_HZ / (f[1] - f[0])) | 1)
    return np.clip(db - uniform_filter1d(db, width, axis=1, mode="nearest"), 0, CLIP_DB)


class HarmonicTracker:
    """Comb + Viterbi pitch tracker for one microphone. All grids are fixed at construction."""

    def __init__(self, fs=8000):
        self.fs = fs
        f = np.fft.rfftfreq(NFFT, 1 / fs)
        self.f = f[(f >= FMIN) & (f <= min(FMAX, 0.49 * fs))]
        fmax = self.f[-1]
        self.lg = np.exp(np.arange(np.log(FMIN), np.log(fmax), LOG_STEP))         # log grid (G,)
        pos = (self.lg - self.f[0]) / (self.f[1] - self.f[0])                       # fractional bin per cell
        self.i0 = np.clip(np.floor(pos).astype(int), 0, len(self.f) - 2)
        self.frac = (pos - self.i0).astype(np.float32)
        lo, hi = np.searchsorted(self.lg, [F0_LO, F0_HI])
        self.cand = np.arange(lo, hi)                                               # candidate f0 cells
        self.f0 = self.lg[self.cand]
        # comb offsets in cells; K per candidate, padded with -1 (masked)
        K = np.clip(np.floor(fmax / self.f0).astype(int), 0, K_MAX)
        self.ok = K >= K_MIN
        k = np.arange(1, K_MAX + 1)
        tooth = np.round(np.log(k) / LOG_STEP).astype(int)
        gap = np.round(np.log(k - 0.5) / LOG_STEP).astype(int)
        valid = k[None] <= K[:, None]
        self.teeth = np.where(valid, self.cand[:, None] + tooth[None], -1)
        self.gaps = np.where(valid, self.cand[:, None] + gap[None], -1)
        w = np.where(valid, HARM_DECAY ** (k[None] - 1.0), 0.0)
        self.w = (w / np.maximum(w.sum(1, keepdims=True), 1e-9)).astype(np.float32)      # (N, K_MAX), rows sum to 1
        self.notch_cells = np.maximum(np.round(np.log1p(NOTCH_HZ / self.lg) / LOG_STEP).astype(int), 3)
        self.jump = round(MAX_JUMP / LOG_STEP)

    # --- 2-4: whitened log-frequency spectrum and comb score ---------------------------------
    def log_spec(self, S):
        """Linear-bin whitened spectrum (F, bins) -> sampled on the log grid (F, G). Only a +-1 bin max:
        wider max-filters raise the noise at the gaps as much as at the teeth (tested: 3-8 dB)."""
        S = maximum_filter1d(S, 2 * TOOTH_BINS + 1, axis=1, mode="nearest")
        return S[:, self.i0] * (1 - self.frac) + S[:, self.i0 + 1] * self.frac

    def comb(self, L):
        """Comb score (F, N candidates) in dB."""
        Lp = np.concatenate([L, np.zeros((len(L), 1), L.dtype)], 1)               # index -1 -> 0
        out = np.empty((len(L), len(self.cand)), np.float32)
        for s in range(0, len(self.cand), 512):
            t, g, w = self.teeth[s: s + 512], self.gaps[s: s + 512], self.w[s: s + 512]
            out[:, s: s + 512] = ((Lp[:, t] - GAP_WEIGHT * Lp[:, g]) * w).sum(2)
        out[:, ~self.ok] = -CLIP_DB
        return out

    # --- 5: Viterbi with a no-tone state --------------------------------------------------------
    def viterbi(self, score):
        """Best path: candidate index per frame, -1 where there is no tone."""
        F, N = score.shape
        J = self.jump
        cost = JUMP_COST * (np.arange(-J, J + 1) / J) ** 2                         # window offset o -> src = n + o - J
        back = np.empty((F, N + 1), np.int32)
        prev = np.r_[np.full(N, -SWITCH), 0.0]                                      # start in "no tone"
        idx = np.arange(N)
        for i in range(F):
            win = np.lib.stride_tricks.sliding_window_view(np.pad(prev[:N], J, constant_values=-np.inf), 2 * J + 1)
            o = np.argmax(win - cost, 1)                                            # tone -> tone
            best = win[idx, o] - cost[o]
            arg = (idx + o - J).astype(np.int32)
            from_null = prev[N] - SWITCH
            better = from_null > best
            best[better], arg[better] = from_null, N
            j = int(np.argmax(prev[:N]))
            null_from_tone = prev[j] - SWITCH
            if prev[N] >= null_from_tone:
                null, null_arg = prev[N], N
            else:
                null, null_arg = null_from_tone, j
            back[i, :N], back[i, N] = arg, null_arg
            prev = np.r_[best + score[i], null + NULL_SCORE]
        path = np.empty(F, np.int32)
        state = int(np.argmax(prev))
        for i in range(F - 1, -1, -1):
            path[i] = state
            state = back[i, state]
        return np.where(path == N, -1, path)

    # --- 6-7: notch and refine -------------------------------------------------------------------
    def notch(self, L, path):
        for i in np.flatnonzero(path >= 0):
            t = self.teeth[path[i]]
            t = t[t >= 0]
            for c in t:
                w = self.notch_cells[c]
                L[i, max(c - w, 0): c + w + 1] = 0

    def refine(self, db, S, f0):
        """Weighted LS pitch from the harmonics' interpolated peaks. Returns (f0_fine, n harmonics used)."""
        df = self.f[1] - self.f[0]
        out, n_used = np.full(len(f0), np.nan), np.zeros(len(f0), int)
        for i in np.flatnonzero(np.isfinite(f0)):
            num = den = 0.0
            for k in range(1, K_MAX + 1):
                fk = k * f0[i]
                if fk > self.f[-1] - 2 * df:
                    break
                half = max(REFINE_REL * fk, 3 * df)                                    # the comb is already ~0.1 % right
                lo, hi = np.searchsorted(self.f, [fk - half, fk + half + df])
                j = lo + int(np.argmax(S[i, lo:hi]))
                if S[i, j] < REFINE_DB or j == 0 or j >= len(self.f) - 1:
                    continue
                a, b, c = db[i, j - 1], db[i, j], db[i, j + 1]
                d = 0.5 * (a - c) / (a - 2 * b + c) if (b >= a and b >= c and a - 2 * b + c < 0) else 0.0
                d = min(max(d, -0.5), 0.5)                     # the peak was picked on S, so db may be flat here
                w = S[i, j]
                num += w * k * (self.f[j] + d * df)
                den += w * k * k
                n_used[i] += 1
            if den > 0 and abs(num / den / f0[i] - 1) <= REFINE_REL:            # else keep the coarse pitch
                out[i] = num / den
        return out, n_used

    def run(self, x, t_centres, n_tones=2):
        """x: one channel at self.fs. Returns dict: f0 (F, n_tones) coarse, f0_fine (F, n_tones),
        score (F, n_tones) comb score on the path, n_harm (F, n_tones); NaN / 0 where there is no tone."""
        db, _ = spectrogram(x, self.fs, t_centres)
        S = whiten(db, self.f)
        L = self.log_spec(S)
        F = len(db)
        res = {k: np.full((F, n_tones), np.nan) for k in ("f0", "f0_fine", "score")}
        res["n_harm"] = np.zeros((F, n_tones), int)
        for m in range(n_tones):
            sc = self.comb(L)
            path = self.viterbi(sc)
            on = path >= 0
            if not on.any():
                break
            res["f0"][on, m] = self.f0[path[on]]
            res["score"][on, m] = sc[np.flatnonzero(on), path[on]]
            res["f0_fine"][:, m], res["n_harm"][:, m] = self.refine(db, S, res["f0"][:, m])
            self.notch(L, path)
        return res


# --- Doppler between clusters --------------------------------------------------------------------

def align(f, ref, tol=0.03):
    """Ratio r in RATIOS with f / r closest to ref (relative), or nan if none is within tol."""
    rel = np.log(f / (RATIOS * ref))
    i = int(np.argmin(np.abs(rel)))
    return RATIOS[i] if abs(rel[i]) < tol else np.nan


def rdot_difference(f_i, f_j):
    """rdot_i - rdot_j (m/s) from the pitch heard at two clusters at the same time, after aligning
    the harmonic numbers. Positive: the drone is moving away from i faster than from j."""
    r = align(f_i, f_j)
    return -C * np.log(f_i / (r * f_j)) if np.isfinite(r) else np.nan
