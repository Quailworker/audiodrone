"""Source-time sound generators. Each returns a float32 signal (arbitrary units) at rate fs.

Physics and parameter ranges: design.md sections 4.1-4.4. Levels are set by the caller.
"""
import numpy as np
from scipy import fft as sfft
from scipy import signal

TABLE = 4096
F_LIMIT = 7500.0          # highest harmonic kept in wavetables (keeps < Nyquist at 2x pitch)


# --- helpers ---------------------------------------------------------------

def smooth_noise(rng, n, fs, cutoff_hz, std=1.0):
    """Gaussian noise low-passed at cutoff_hz, rescaled to the given std."""
    step = max(1, int(fs / (cutoff_hz * 8)))
    coarse = rng.standard_normal(n // step + 4)
    b, a = signal.butter(2, min(0.99, cutoff_hz / (fs / step / 2)))
    coarse = signal.filtfilt(b, a, coarse)
    coarse /= coarse.std() + 1e-12
    return (np.interp(np.arange(n), np.arange(len(coarse)) * step, coarse) * std).astype(np.float32)


def colored_noise(rng, n, exponent=1.0, f_lo=None, f_hi=None, fs=32000):
    """Noise with power spectrum ~ 1/f^exponent, optionally band-limited."""
    m = sfft.next_fast_len(n, real=True)
    X = rng.standard_normal(m // 2 + 1) + 1j * rng.standard_normal(m // 2 + 1)
    f = sfft.rfftfreq(m, 1 / fs)
    g = 1 / np.maximum(f, 10.0) ** (exponent / 2)
    if f_lo:
        g *= 1 / (1 + (f_lo / np.maximum(f, 1e-3)) ** 4)
    if f_hi:
        g *= 1 / (1 + (f / f_hi) ** 4)
    x = sfft.irfft(X * g, m)[:n]
    return (x / (x.std() + 1e-12)).astype(np.float32)


def harmonic_table(amps, phases):
    """One period of sum_k amps[k] cos(k*theta + phases[k]), sampled on TABLE points."""
    th = np.arange(TABLE) * 2 * np.pi / TABLE
    k = np.arange(1, len(amps) + 1)[:, None]
    return (amps[:, None] * np.cos(k * th + phases[:, None])).sum(0).astype(np.float32)


def table_lookup(table, phase_cycles):
    """Read a periodic table at phase (in cycles) with linear interpolation."""
    pos = (phase_cycles % 1.0) * TABLE
    i = pos.astype(np.int64)
    d = (pos - i).astype(np.float32)
    return table[i] * (1 - d) + table[(i + 1) % TABLE] * d


def random_harmonic_amps(rng, n, slope, jitter_db=4.0):
    k = np.arange(1, n + 1)
    return (k ** -slope * 10 ** (rng.normal(0, jitter_db, n) / 20)).astype(np.float32)


def phase_cycles(freq_hz, fs):
    return np.cumsum(freq_hz.astype(np.float64)) / fs


# --- electric multirotor ------------------------------------------------------

def electric_rotors(rng, shaft_hz, fs, n_blades, pole_pairs, p):
    """Sum of rotors. shaft_hz: (n_rotors, N) shaft rate per rotor in Hz.

    p: dict of per-drone sound parameters (see scene.sample_electric).
    """
    n_rot, n = shaft_hz.shape
    bpf_max = n_blades * shaft_hz.max()
    k_bpf = max(1, int(F_LIMIT / bpf_max))
    bpf_table = harmonic_table(random_harmonic_amps(rng, k_bpf, p["slope"]), rng.uniform(0, 2 * np.pi, k_bpf))
    # shaft-rate lines from blade/rotor imbalance (non-blade orders only)
    k_sh = min(int(F_LIMIT / shaft_hz.max()), 3 * n_blades)
    sh_amps = random_harmonic_amps(rng, k_sh, 1.0) * p["imbalance"]
    sh_amps[n_blades - 1::n_blades] = 0
    sh_table = harmonic_table(sh_amps, rng.uniform(0, 2 * np.pi, k_sh))
    k_wh = max(1, min(4, int(F_LIMIT / (pole_pairs * shaft_hz.max()))))
    wh_table = harmonic_table(random_harmonic_amps(rng, k_wh, 1.5), rng.uniform(0, 2 * np.pi, k_wh))

    out = np.zeros(n, np.float32)
    for i in range(n_rot):
        # turbulence: slow phase wander (broadens high harmonics more) and gain flutter
        wander = smooth_noise(rng, n, fs, 3.0, p["phase_wander"]) / (2 * np.pi)
        cyc = phase_cycles(shaft_hz[i], fs) + rng.uniform() + wander
        gain = 10 ** (smooth_noise(rng, n, fs, 2.0, p["flutter_db"]) / 20)
        tone = table_lookup(bpf_table, n_blades * cyc)
        tone += table_lookup(sh_table, cyc)
        tone += p["whine"] * table_lookup(wh_table, pole_pairs * cyc)
        bb = colored_noise(rng, n, 1.0, 150, 6000, fs)
        bb *= 1 + p["bb_mod"] * np.cos(2 * np.pi * n_blades * cyc).astype(np.float32)
        out += gain * (tone + p["broadband"] * bb)
    if p.get("pwm_hz"):
        out += p["pwm"] * np.sin(2 * np.pi * p["pwm_hz"] * np.arange(n) / fs).astype(np.float32)
    return out


# --- combustion engines --------------------------------------------------------

def pulse_kernel(rng, fs, res_hz, q, click, muffled):
    """Exhaust pulse: pipe resonance + sharp transient, optionally low-passed by a muffler."""
    t = np.arange(int(0.04 * fs)) / fs
    k = np.exp(-np.pi * res_hz / q * t) * np.sin(2 * np.pi * res_hz * t)
    k += click * np.exp(-t / 3e-4) * rng.standard_normal(len(t)) * 0.5
    if muffled:
        b, a = signal.butter(2, muffled / (fs / 2))
        k = signal.lfilter(b, a, k)
    return (k / np.abs(k).max()).astype(np.float32)


def engine(rng, shaft_hz, fs, fires_per_rev, p):
    """Combustion engine as a jittered pulse train convolved with an exhaust pulse.

    fires_per_rev: 1 for a single-cylinder two-stroke, 0.5 for a single-cylinder four-stroke,
    n_cyl/2 for a four-stroke multi-cylinder.
    """
    n = len(shaft_hz)
    cyc = phase_cycles(shaft_hz * fires_per_rev, fs) + rng.uniform()
    idx = np.flatnonzero(np.diff(np.floor(cyc)) > 0)
    period = fs / np.maximum(shaft_hz[idx] * fires_per_rev, 1.0)
    pos = idx + rng.normal(0, p["jitter"], len(idx)) * period
    amp = np.clip(rng.normal(1, p["cycle_var"], len(idx)), 0.2, 2.0)
    amp[rng.uniform(size=len(idx)) < p["misfire"]] *= 0.05
    if p.get("cyl_imbalance"):               # uneven cylinders create half-order lines
        amp *= 1 + p["cyl_imbalance"] * (np.arange(len(idx)) % 2)
    train = np.zeros(n, np.float32)
    pos = np.clip(np.round(pos).astype(np.int64), 0, n - 1)
    np.add.at(train, pos, amp.astype(np.float32))
    kern = pulse_kernel(rng, fs, p["res_hz"], p["q"], p["click"], p["muffler_hz"])
    out = signal.fftconvolve(train, kern)[:n].astype(np.float32)
    out /= out.std() + 1e-12
    mech = colored_noise(rng, n, 0.5, 800, 9000, fs) * p["mech"]
    return out + mech


def propeller(rng, shaft_hz, fs, n_blades, slope):
    k = max(1, int(F_LIMIT / (n_blades * shaft_hz.max())))
    table = harmonic_table(random_harmonic_amps(rng, k, slope), rng.uniform(0, 2 * np.pi, k))
    wander = smooth_noise(rng, len(shaft_hz), fs, 3.0, 0.3) / (2 * np.pi)
    return table_lookup(table, n_blades * (phase_cycles(shaft_hz, fs) + wander))


# --- other sources ---------------------------------------------------------------

def fan(rng, n, fs):
    rpm = rng.uniform(500, 1500)
    blades = int(rng.integers(3, 10))
    shaft = np.full(n, rpm / 60, np.float32) * (1 + smooth_noise(rng, n, fs, 0.2, 0.003))
    tone = propeller(rng, shaft, fs, blades, rng.uniform(1.0, 2.0))
    return tone + rng.uniform(0.3, 1.5) * colored_noise(rng, n, 1.0, 60, 4000, fs)


def mains_hum(rng, n, fs):
    base = rng.choice([50.0, 60.0])
    amps = random_harmonic_amps(rng, 12, rng.uniform(0.5, 1.5), 8)
    if rng.uniform() < 0.5:                  # transformer: magnetostriction at 2x mains
        amps[0] *= 0.1
    f = base * (1 + smooth_noise(rng, n, fs, 0.1, 0.0005))
    return table_lookup(harmonic_table(amps, rng.uniform(0, 2 * np.pi, 12)), phase_cycles(f, fs))


def insect(rng, n, fs):
    wing = rng.uniform(150, 650)
    f = wing * (1 + smooth_noise(rng, n, fs, rng.uniform(2, 10), rng.uniform(0.01, 0.05)))
    tone = table_lookup(harmonic_table(random_harmonic_amps(rng, 10, rng.uniform(0.7, 1.5)),
                                       rng.uniform(0, 2 * np.pi, 10)), phase_cycles(f, fs))
    return tone * 10 ** (smooth_noise(rng, n, fs, 1.0, 4) / 20)


def birds(rng, n, fs):
    out = np.zeros(n, np.float32)
    t = 0.0
    while True:
        t += rng.exponential(2.0)
        start = int(t * fs)
        if start >= n:
            return out
        for _ in range(int(rng.integers(1, 6))):
            dur = rng.uniform(0.04, 0.3)
            m = int(dur * fs)
            tt = np.arange(m) / fs
            f0, f1 = rng.uniform(1500, 8000, 2)
            ph = 2 * np.pi * (f0 * tt + (f1 - f0) * tt ** 2 / (2 * dur))
            chirp = np.sin(ph) * np.hanning(m)
            end = min(n, start + m)
            out[start:end] += chirp[: end - start]
            start = end + int(rng.uniform(0.02, 0.15) * fs)
            if start >= n:
                return out


def crickets(rng, n, fs):
    carrier = np.sin(2 * np.pi * rng.uniform(3000, 6000) * np.arange(n) / fs)
    pulse = (np.sin(2 * np.pi * rng.uniform(15, 60) * np.arange(n) / fs) > 0.3).astype(np.float32)
    chirp_gate = (np.sin(2 * np.pi * rng.uniform(0.3, 2) * np.arange(n) / fs) > 0).astype(np.float32)
    return (carrier * pulse * chirp_gate).astype(np.float32)


def tyre_noise(rng, n, fs):
    return colored_noise(rng, n, 0.5, 300, 3000, fs)


# --- diffuse beds and mic-local noise ------------------------------------------------

def bed(rng, kind, n, fs):
    if kind == "wind_trees":
        x = colored_noise(rng, n, 1.0, 80, 5000, fs)
        return x * 10 ** (smooth_noise(rng, n, fs, 0.3, 5) / 20)
    if kind == "traffic_rumble":
        x = colored_noise(rng, n, 2.0, 25, 1500, fs)
        return x * 10 ** (smooth_noise(rng, n, fs, 0.1, 4) / 20)
    if kind == "rain":
        x = colored_noise(rng, n, 0.0, 800, 12000, fs)
        drops = (rng.uniform(size=n) < 2e-4) * rng.standard_normal(n) * 8
        return (x + drops).astype(np.float32)
    if kind == "urban":
        return colored_noise(rng, n, 1.3, 40, 6000, fs)
    if kind == "quiet_rural":
        return colored_noise(rng, n, 1.0, 30, 8000, fs)
    raise ValueError(kind)


def mic_wind(rng, n, fs):
    """Turbulence at the windshield: strong below ~200 Hz, gusty, independent per mic."""
    x = colored_noise(rng, n, 2.0, 5, rng.uniform(80, 300), fs)
    return x * 10 ** (smooth_noise(rng, n, fs, 0.5, 8) / 20)
