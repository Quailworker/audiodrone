"""Outdoor propagation: emission time, fractional delay, spreading, ground echo, air absorption.

All signals are in pascals. Design reference: design.md section 14.6.
"""
import numpy as np
from scipy import fft as sfft
from scipy import signal

C_SOUND = 343.0
P_REF = 20e-6
FULL_SCALE_DB = 120.0
FULL_SCALE_PA = P_REF * 10 ** (FULL_SCALE_DB / 20)


def interp_path(t_ctrl, p_ctrl, t):
    """Linearly interpolate a (N, 3) control-rate path at times t."""
    return np.stack([np.interp(t, t_ctrl, p_ctrl[:, i]) for i in range(3)], axis=-1)


def emission_time(t, t_ctrl, p_ctrl, mic, iters=5):
    """Solve t = tau + |p(tau) - mic| / c for tau (fixed point, contracts by |v|/c)."""
    tau = t.copy()
    for _ in range(iters):
        r = np.linalg.norm(interp_path(t_ctrl, p_ctrl, tau) - mic, axis=-1)
        tau = t - r / C_SOUND
    return tau, r


def read_fractional(s, pos):
    """4-point cubic Lagrange read of s at fractional sample positions pos.

    Positions outside the buffer (sound emitted before the source history starts) read as silence.
    """
    i = np.floor(pos).astype(np.int64)
    d = (pos - i).astype(np.float32)
    inside = (i >= 1) & (i <= len(s) - 3)
    i = np.clip(i, 1, len(s) - 3)
    xm1, x0, x1, x2 = s[i - 1], s[i], s[i + 1], s[i + 2]
    y = (-d * (d - 1) * (d - 2) / 6 * xm1
         + (d + 1) * (d - 1) * (d - 2) / 2 * x0
         - (d + 1) * d * (d - 2) / 2 * x1
         + (d + 1) * d * (d - 1) / 6 * x2)
    return np.where(inside, y, 0).astype(np.float32)


def air_absorption_db_per_m(f, temp_c=20.0, rh=70.0, pa_kpa=101.325):
    """ISO 9613-1 pure-tone atmospheric absorption coefficient (dB/m)."""
    T, T0, T01 = temp_c + 273.15, 293.15, 273.16
    pr = pa_kpa / 101.325
    h = rh * 10 ** (-6.8346 * (T01 / T) ** 1.261 + 4.6151) / pr
    fr_o = pr * (24 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    fr_n = pr * (T / T0) ** -0.5 * (9 + 280 * h * np.exp(-4.170 * ((T / T0) ** (-1 / 3) - 1)))
    f2 = f ** 2
    return 8.686 * f2 * (1.84e-11 / pr * (T / T0) ** 0.5 + (T / T0) ** -2.5 * (
        0.01275 * np.exp(-2239.1 / T) / (fr_o + f2 / fr_o)
        + 0.1068 * np.exp(-3352.0 / T) / (fr_n + f2 / fr_n)))


def apply_air_absorption(x, r, fs, temp_c, rh, nfft=1024):
    """Time-varying absorption: per STFT frame gain 10^(-alpha(f) * r / 20)."""
    hop = nfft // 4
    f, t, X = signal.stft(x, fs, nperseg=nfft, noverlap=nfft - hop)
    r_frames = np.interp(t, np.arange(len(r)) / fs, r)
    alpha = air_absorption_db_per_m(f, temp_c, rh)
    X *= 10 ** (-np.outer(alpha, r_frames) / 20)
    _, y = signal.istft(X, fs, nperseg=nfft, noverlap=nfft - hop)
    return y[: len(x)].astype(np.float32)


def a_weighting_db(f):
    f2 = np.maximum(np.asarray(f, float), 1e-3) ** 2
    ra = (12194 ** 2 * f2 ** 2) / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194 ** 2))
    return 20 * np.log10(ra) + 2.0


def a_weighted_rms(x, fs):
    m = sfft.next_fast_len(len(x), real=True)
    X = sfft.rfft(x, m)
    w = 10 ** (a_weighting_db(sfft.rfftfreq(m, 1 / fs)) / 20)
    return np.sqrt(2 * np.sum(np.abs(X * w) ** 2) / (m * len(x)))


def set_level_dba(x, fs, level_db):
    """Scale x so its A-weighted level is level_db SPL (x in arbitrary units -> pascals)."""
    rms = a_weighted_rms(x, fs)
    return (x * (P_REF * 10 ** (level_db / 20) / max(rms, 1e-20))).astype(np.float32)


def fractional_shift(x, delay_s, fs):
    """Circularly delay x by a constant (fractional) time via FFT phase."""
    if abs(delay_s) * fs < 1e-3:
        return x
    m = sfft.next_fast_len(len(x), real=True)
    X = sfft.rfft(x, m)
    X *= np.exp(-2j * np.pi * sfft.rfftfreq(m, 1 / fs) * delay_s)
    return sfft.irfft(X, m)[: len(x)].astype(np.float32)


def propagate_point_source(src, src_t0, fs, t_ctrl, p_ctrl, mic, n_out, ground_r, temp_c, rh,
                           directivity=None):
    """Signal at `mic` (pascals) from a moving point source with level-at-1-m signal `src`.

    src[n] is the source pressure referred to 1 m at source time src_t0 + n/fs.
    Returns (signal, r_direct sampled at output rate).
    """
    t = np.arange(n_out) / fs
    # emission time on a 1 kHz grid, interpolated to audio rate (tau is smooth)
    t_c = np.arange(0, n_out / fs + 1e-3, 1e-3)
    out = np.zeros(n_out, np.float32)
    r_direct = None
    for image in (False, True):
        m = mic * np.array([1, 1, -1]) if image else mic   # image source == mirrored mic
        tau_c, r_c = emission_time(t_c, t_ctrl, p_ctrl, m)
        tau = np.interp(t, t_c, tau_c)
        r = np.interp(t, t_c, r_c).astype(np.float32)
        y = read_fractional(src, (tau - src_t0) * fs) / np.maximum(r, 0.5)
        if directivity is not None:
            p = interp_path(t_ctrl, p_ctrl, tau_c)
            el = np.arcsin(np.clip((p[:, 2] - m[2]) / np.maximum(r_c, 1e-3), -1, 1))
            y *= np.interp(t, t_c, directivity(tau_c, el)).astype(np.float32)
        if image:
            out += ground_r * y
        else:
            out += y
            r_direct = r
    return apply_air_absorption(out, r_direct, fs, temp_c, rh), r_direct
