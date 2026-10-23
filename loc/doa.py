"""Direction finding per cluster (tasks/phase1_localisation.md step 2; design.md 7.3-7.4, 7.8, 9.6).

Three estimators on the same per-frame spectra (Hann frames of win_s, band-limited):

  srp  SRP-PHAT-beta: per mic pair the weighted cross-spectrum G_ij / |G_ij|^beta, steered to every
       direction of a coarse grid (2 deg) and summed over the 6 pairs, then refined on a local grid
       (0.25 deg, +-2 deg). Evaluated exactly at each direction (no lag-grid interpolation).
  ls   least squares on the 6 pair delays: each pair's weighted GCC on a lag grid, its peak inside the
       pair's physical lag range (parabolic interpolation), then -(p_i - p_j).u = c tau_ij solved for u.
       Cheap; the ESP32 option (mode B, design 9.6).
  bf   reflection-aware steered beamformer: the steering vector includes the ground echo,
       a_i = exp(jk(x_i u_x + y_i u_y)) (exp(jk z_i u_z) + R exp(-jk z_i u_z)), z_i the mic height above
       the ground, |a| = 1, input X / |X|^beta per frequency. Pairwise methods cannot model the echo: on
       a 2 m pole the echo of a tone arrives at the same frequencies from the mirror direction and biases
       the pair delays (3-15 deg noise-free), and on the ground the echo cancels the vertical phase at a
       tetrahedron's top mic (10-40 deg), while bf stays within 0.1-2 deg (design 13.5).

Weighting: beta = 0.5 by default. Full PHAT (beta = 1) whitens every bin, so with tonal drone sound the
many noise-only bins swamp the few tone bins (flaw F4): on a synthetic 12-harmonic comb at 30 dB SNR,
beta = 1 gave a 4-5 deg median error, beta = 0.5 under 0.1 deg (tests/test_doa.py).

Conventions: u is the unit vector from the cluster towards the source; az from +x towards +y, el above
the horizontal. A plane wave from u reaches mic i at t_i = -p_i.u / c, so the pair delay is
tau_ij(u) = t_i - t_j = -(p_i - p_j).u / c and the cross-spectrum X_i X_j* has phase -2 pi f tau_ij.
Flat clusters measure only the horizontal part of u; their grids start at el = 0 (nothing flies below
the ground) and ls takes u_z = +sqrt(1 - |u_h|^2).

The reference implementation runs in torch (GPU if present); the per-frame work is fixed-size, and
ops_per_frame() gives the multiply-add budget of each option for the ESP32 question.
"""
import itertools

import numpy as np
import torch

C = 343.0


def _unit(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], -1)


def _grid(step, el_lo):
    az = np.arange(0, 360, step)
    el = np.arange(el_lo, 90 + 1e-9, step)
    A, E = np.meshgrid(az, el, indexing="ij")
    keep = (E < 90) | (A == 0)                                                   # one point at the zenith
    A, E = A[keep], E[keep]
    return {"az": A, "el": E, "u": _unit(A, E)}


def angle_deg(u, v):
    """Angle between unit vectors (..., 3)."""
    return np.degrees(np.arccos(np.clip(np.sum(u * v, -1), -1, 1)))


def _result(u, coh, **extra):
    u = u.astype(float)
    return {"u": u, "az": np.degrees(np.arctan2(u[:, 1], u[:, 0])) % 360,
            "el": np.degrees(np.arcsin(np.clip(u[:, 2], -1, 1))), "coh": coh, **extra}


class DOA:
    """Fixed-size direction finder for one cluster geometry (methods srp and ls).

    mics: (M, 3) offsets from the cluster centre. fs: sample rate. win_s: frame length (s).
    band: (lo, hi) Hz used for direction finding. lag_step: GCC lag-grid step (s) for ls.
    """

    def __init__(self, mics, fs, win_s=0.4, band=(100.0, 860.0), lag_step=2e-6, coarse_deg=2.0,
                 fine_deg=0.25, fine_span_deg=2.0, phat_beta=0.5, flat=None, device=None):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mics = np.asarray(mics, float)
        self.pairs = list(itertools.combinations(range(len(self.mics)), 2))
        self.fs = fs
        self.win = int(round(win_s * fs))
        self.nfft = 1 << (self.win - 1).bit_length()
        f = np.fft.rfftfreq(self.nfft, 1 / fs)
        self.sel = (f >= band[0]) & (f <= band[1])
        self.f = f[self.sel]
        self.phat_beta = phat_beta
        self.window = np.hanning(self.win).astype(np.float32)
        self.flat = bool(np.ptp(self.mics[:, 2]) < 1e-6) if flat is None else flat
        self.d = np.array([self.mics[i] - self.mics[j] for i, j in self.pairs])   # (P, 3)
        self.max_lag = np.linalg.norm(self.d, axis=1) / C
        L = self.max_lag.max() * 1.05
        self.lags = np.arange(-L, L + lag_step, lag_step)
        self.lag_step = lag_step
        self.el_lo = 0.0 if self.flat else -15.0
        self.coarse_deg, self.fine_deg, self.fine_span = coarse_deg, fine_deg, fine_span_deg
        self.grid = _grid(coarse_deg, self.el_lo)
        self.w = torch.tensor(2 * np.pi * self.f, dtype=torch.float32, device=self.dev)
        self.dt = torch.tensor(self.d / C, dtype=torch.float32, device=self.dev)
        self.gu = torch.tensor(self.grid["u"], dtype=torch.float32, device=self.dev)
        # cache the coarse steering table only when it is small (8 GB GPUs run several of these at once)
        cache = len(self.pairs) * len(self.f) * len(self.gu) * 8 < 256e6
        self.Eg = self._pair_steer(self.gu) if cache else None
        lags = torch.tensor(self.lags, dtype=torch.float32, device=self.dev)
        self.Elag = torch.exp(1j * self.w[:, None] * lags[None])                   # (B, L)
        off = np.arange(-fine_span_deg, fine_span_deg + 1e-9, fine_deg)
        self.fine_off = [g.ravel() for g in np.meshgrid(off, off, indexing="ij")]

    def _pair_steer(self, u):
        """u (..., 3) -> exp(+j w tau_p(u)), shape (P, B, ...): the phase that cancels G_p's phase."""
        tau = -(u @ self.dt.T)                                                    # (..., P)
        return torch.exp(1j * self.w.view(-1, *([1] * tau.dim())) * tau).movedim(-1, 0)

    # --- spectra -------------------------------------------------------------------------
    def frames(self, x, t_centres):
        """x: (M, n) audio. Returns (F, M, B) complex band-limited spectra (torch) of Hann frames."""
        start = np.round(np.asarray(t_centres) * self.fs).astype(int) - self.win // 2
        idx = start[:, None] + np.arange(self.win)[None]
        valid = (idx >= 0) & (idx < x.shape[1])
        seg = np.where(valid[:, None, :], x[:, np.clip(idx, 0, x.shape[1] - 1)].transpose(1, 0, 2), 0)
        seg = torch.tensor(seg * self.window, dtype=torch.float32, device=self.dev)
        return torch.fft.rfft(seg, self.nfft, dim=-1)[..., torch.tensor(self.sel, device=self.dev)]

    def weighted_cross(self, X):
        """(F, P, B) beta-weighted cross-spectra, normalised so that a perfect match sums to 1 per pair."""
        G = torch.stack([X[:, i] * X[:, j].conj() for i, j in self.pairs], 1)
        G = G / G.abs().clamp_min(1e-20) ** self.phat_beta
        return G / G.abs().sum(-1, keepdim=True).clamp_min(1e-30)

    # --- estimators ----------------------------------------------------------------------
    def srp(self, G):
        F, P, _ = G.shape
        Gp = G.permute(1, 0, 2)                                                   # (P, F, B)
        if self.Eg is not None:
            m = torch.bmm(Gp, self.Eg).real.sum(0) / P                            # (F, Grid)
        else:
            m = torch.cat([torch.bmm(Gp, self._pair_steer(self.gu[i: i + 1024])).real.sum(0) / P
                           for i in range(0, len(self.gu), 1024)], 1)
        best = m.argmax(1)
        Fi = torch.arange(F, device=self.dev)
        peak_c = m[Fi, best]
        gu = self.gu
        far = (gu[best] @ gu.T) < np.cos(np.radians(20))
        second = torch.where(far, m, torch.full_like(m, -1e9)).max(1).values
        best = best.cpu().numpy()
        az = self.grid["az"][best][:, None] + self.fine_off[0][None]
        el = np.clip(self.grid["el"][best][:, None] + self.fine_off[1][None], self.el_lo, 90)
        u = torch.tensor(_unit(az, el), dtype=torch.float32, device=self.dev)    # (F, K, 3)
        mf = torch.einsum("fpb,pbfk->fk", G, self._pair_steer(u)).real / P
        k = mf.argmax(1)
        return _result(u[Fi, k].cpu().numpy(), mf[Fi, k].cpu().numpy(),
                       ratio=(peak_c / second.clamp_min(1e-6)).cpu().numpy())

    def ls(self, G):
        r = (G @ self.Elag).real.cpu().numpy()                                    # (F, P, L)
        F, P, _ = r.shape
        tau = np.zeros((F, P))
        peak = np.zeros((F, P))
        rows = np.arange(F)
        for p in range(P):
            idx = np.flatnonzero(np.abs(self.lags) <= self.max_lag[p] + self.lag_step)
            rp = r[:, p, idx]
            j = rp.argmax(1)
            a = rp[rows, np.clip(j - 1, 0, len(idx) - 1)]
            b = rp[rows, j]
            c = rp[rows, np.clip(j + 1, 0, len(idx) - 1)]
            den = a - 2 * b + c
            safe = np.abs(den) > 1e-12
            frac = np.clip(np.where(safe, 0.5 * (a - c) / np.where(safe, den, 1), 0), -0.5, 0.5)
            tau[:, p] = self.lags[idx[j]] + frac * self.lag_step
            peak[:, p] = b
        A = -self.d / C                                                          # tau = A u
        if self.flat:
            sol = np.linalg.lstsq(A[:, :2], tau.T, rcond=None)[0].T              # (F, 2)
            h = np.linalg.norm(sol, axis=1)
            sol = np.where(h[:, None] > 1, sol / np.maximum(h, 1e-9)[:, None], sol)
            u = np.column_stack([sol, np.sqrt(np.clip(1 - np.sum(sol ** 2, 1), 0, 1))])
        else:
            u = np.linalg.lstsq(A, tau.T, rcond=None)[0].T
            u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
        return _result(u, peak.mean(1))

    @torch.no_grad()
    def run(self, x, t_centres, methods=("srp", "ls"), chunk=64):
        out = {m: [] for m in methods}
        for s in range(0, len(t_centres), chunk):
            G = self.weighted_cross(self.frames(x, t_centres[s: s + chunk]))
            for m in methods:
                out[m].append(getattr(self, m)(G))
        return {m: {k: np.concatenate([d[k] for d in v]) for k in v[0]} for m, v in out.items()}

    def ops_per_frame(self):
        """Approximate real multiply-adds per frame for each option (for the ESP32 budget, design 9.6).
        GCC via one inverse FFT per pair (the embedded way) rather than the lag-grid product used here."""
        M, P, B, G = len(self.mics), len(self.pairs), len(self.f), len(self.grid["az"])
        K = len(self.fine_off[0])
        fft = M * 2.5 * self.nfft * np.log2(self.nfft)                           # real FFTs of the frames
        cross = P * B * 8                                                        # X_i X_j*, |.|^beta, normalise
        ifft = P * 2.5 * self.nfft * np.log2(self.nfft)                          # GCC per pair
        lookup = 3 * P                                                           # interpolate 6 GCCs per direction
        return {"nfft": self.nfft, "n_bins": B, "n_grid": G, "n_fine": K,
                "fft": fft, "cross_spectra": cross, "gcc_ifft": ifft,
                "ls": fft + cross + ifft + P * 40 + 300,
                "srp_coarse_only": fft + cross + ifft + G * lookup,
                "srp": fft + cross + ifft + (G + K) * lookup,
                "srp_exact_steering": fft + cross + (G + K) * P * B * 4,
                "bf": fft + M * B * 3 + (G + K) * M * B * 4}


class BeamformerDOA:
    """Reflection-aware steered response power (method "bf"; see the module docstring).

    mics: (M, 3) offsets from the cluster centre; height: the centre's height above the ground (m);
    refl: assumed ground reflection coefficient R (0 = free field). Same frames, band and beta as DOA.
    """

    def __init__(self, mics, fs, height, refl=0.75, device=None, **kw):
        self.base = DOA(mics, fs, device=device, **kw)
        self.dev = self.base.dev
        self.refl = refl
        xyz = np.asarray(mics, float).copy()
        xyz[:, 2] += height
        self.xyz = torch.tensor(xyz, dtype=torch.float32, device=self.dev)
        self.k = torch.tensor(2 * np.pi * self.base.f / C, dtype=torch.float32, device=self.dev)
        self.grid = _grid(self.base.coarse_deg, 0.0)
        self.gu = torch.tensor(self.grid["u"], dtype=torch.float32, device=self.dev)
        cache = len(self.base.f) * len(xyz) * len(self.gu) * 8 < 256e6         # see DOA: cache only if small
        self.A = self._steer(self.gu) if cache else None

    def _steer(self, u):
        """u (..., 3) -> steering (B, M, ...) complex64, unit norm over mics."""
        h = u[..., :2] @ self.xyz[:, :2].T                                        # (..., M)
        v = u[..., 2:3] * self.xyz[:, 2]                                          # (..., M)
        k = self.k.view(-1, *([1] * h.dim()))
        a = torch.exp(1j * k * h) * (torch.exp(1j * k * v) + self.refl * torch.exp(-1j * k * v))
        a = a / torch.linalg.vector_norm(a, dim=-1, keepdim=True).clamp_min(1e-9)
        return a.movedim(-1, 1)                                                   # (B, M, ...)

    @torch.no_grad()
    def run(self, x, t_centres, chunk=16, weights=None):
        """weights: optional (frames, bins) weights on the band's bins (e.g. a comb mask, run_doa --comb)."""
        b = self.base
        outs = []
        for s in range(0, len(t_centres), chunk):
            X = b.frames(x, t_centres[s: s + chunk])                              # (F, M, B)
            X = X / torch.linalg.vector_norm(X, dim=1, keepdim=True).clamp_min(1e-20) ** b.phat_beta
            if weights is not None:
                X = X * torch.tensor(weights[s: s + chunk], dtype=torch.float32, device=self.dev)[:, None, :]
            energy = (X.abs() ** 2).sum((1, 2)).clamp_min(1e-30)
            Xb = X.permute(2, 0, 1)                                               # (B, F, M)
            if self.A is not None:
                P = (torch.bmm(Xb, self.A.conj()).abs() ** 2).sum(0)              # (F, G)
            else:
                P = torch.cat([(torch.bmm(Xb, self._steer(self.gu[i: i + 1024]).conj()).abs() ** 2).sum(0)
                               for i in range(0, len(self.gu), 1024)], 1)
            best = P.argmax(1).cpu().numpy()
            az = self.grid["az"][best][:, None] + b.fine_off[0][None]
            el = np.clip(self.grid["el"][best][:, None] + b.fine_off[1][None], 0, 90)
            u = torch.tensor(_unit(az, el), dtype=torch.float32, device=self.dev)
            Pf = (torch.einsum("bfm,bmfk->bfk", Xb, self._steer(u).conj()).abs() ** 2).sum(0)
            kk = Pf.argmax(1)
            Fi = torch.arange(len(kk), device=self.dev)
            outs.append(_result(u[Fi, kk].cpu().numpy(), (Pf[Fi, kk] / energy).cpu().numpy()))
        return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
