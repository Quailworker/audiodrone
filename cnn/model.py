"""CNN14 (NeMo/PANNs Cnn14_DecisionLevelMax) with a swappable front end and a drone head.

Front ends (design.md 5.8.4):
  "mel"       - stock log-mel, n_fft 1024 (option A, baseline)
  "logf"      - mel bent at 50 Hz (log-spaced over the whole drone range), n_fft 4096 (option B, model 1)
  "logf_comb" - "logf" plus two frequency-tracker channels: whitened spectrum and harmonic-comb score (model 2)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import LogmelFilterBank, Spectrogram

from .panns_models import Cnn14_DecisionLevelMax, init_bn, init_layer
from .panns_pytorch_utils import interpolate, pad_framewise_output

SR, HOP, N_MELS, FMIN, FMAX = 32000, 320, 64, 50.0, 14000.0
CLASSES = ["drone_electric", "drone_engine", "ground_engine"]
N_CLASSES = len(CLASSES)


def bent_mel_filters(n_fft, sr=SR, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, bend_hz=50.0):
    """Triangular filters evenly spaced on m(f) = log(1 + f/bend_hz), Slaney-normalised.

    Triangles narrower than one FFT bin are widened to one bin each side so no band is empty.
    Returns (n_fft//2 + 1, n_mels), the layout torchlibrosa's melW uses.
    """
    m = np.linspace(np.log1p(fmin / bend_hz), np.log1p(fmax / bend_hz), n_mels + 2)
    edges = bend_hz * np.expm1(m)
    freqs = np.arange(n_fft // 2 + 1) * sr / n_fft
    df = sr / n_fft
    W = np.zeros((len(freqs), n_mels), np.float32)
    for i in range(n_mels):
        c = edges[i + 1]
        lo, hi = min(edges[i], c - df), max(edges[i + 2], c + df)
        up = (freqs - lo) / (c - lo)
        down = (hi - freqs) / (hi - c)
        W[:, i] = np.maximum(0, np.minimum(up, down)) * 2 / (hi - lo)
    return W


# Frequency-tracker channels (design.md 5.2): candidate shaft rates, one per row of the 64-row input.
COMB_F0 = np.geomspace(20.0, 2000.0, N_MELS)
COMB_FMAX, COMB_KMAX, COMB_KMIN, COMB_TOOTH, FLOOR_HZ, SOFT_P = 6000.0, 30, 4, 0.02, 480.0, 4.0


def comb_matrices(n_fft=4096, sr=SR):
    """T (n_fft//2+1, n_teeth): each column averages the FFT bins within +-max(half a bin, 2 % of the centre) of
    harmonic k of one candidate f0 (wider at high k, since rotors differ by a few %), or of the gap (k - 1/2) f0
    between harmonics; G (n_teeth, N_MELS) gives each candidate mean(teeth) - mean(gaps).

    The gaps fix both octave errors: at f0/2 half the teeth land in the true gaps, and at 2 f0 the gaps land on the
    true odd harmonics. Means are over at least COMB_KMIN harmonics so a single loud tone cannot win on its own.
    Harmonics run to COMB_FMAX or COMB_KMAX, whichever comes first."""
    freqs = np.arange(n_fft // 2 + 1) * sr / n_fft
    df = sr / n_fft
    cols, rows = [], []
    for j, f0 in enumerate(COMB_F0):
        ks = [k for k in range(1, COMB_KMAX + 1) if k * f0 <= COMB_FMAX]
        w = 1.0 / max(len(ks), COMB_KMIN)
        for k in ks:
            for fc, sign in ((k * f0, 1.0), ((k - 0.5) * f0, -1.0)):
                sel = np.abs(freqs - fc) <= max(0.5 * df, COMB_TOOTH * fc)
                cols.append(sel / sel.sum())
                rows.append((j, sign * w))
    G = np.zeros((len(rows), N_MELS), np.float32)
    for i, (j, w) in enumerate(rows):
        G[i, j] = w
    return np.stack(cols, 1).astype(np.float32), G


class CombFeatures(nn.Module):
    """Power spectrogram (B, 1, T, F) -> (B, 2, T, 64), fixed (no parameters).

    white = dB above a running 480 Hz mean of the log spectrum, clipped to [0, 40] and scaled to [0, 1], so tones
    stand out and the broadband floor (wind, road) is gone.
      ch 0: white per log-f band, as a power mean (p = 4) so one tone in a wide band still shows
      ch 1: comb score for 64 candidate rates 20-2000 Hz (shaft or blade-pass): mean over harmonics of each
            tooth's power mean, minus the same between harmonics (comb_matrices). Evenly spaced lines light up
            their f0 row; scattered tones and broadband noise do not.
    """

    def __init__(self, melW, n_fft=4096):
        super().__init__()
        W = torch.as_tensor(melW, dtype=torch.float32)
        T, G = comb_matrices(n_fft)
        self.register_buffer("W", W / W.sum(0, keepdim=True).clamp(min=1e-8), persistent=False)
        self.register_buffer("T", torch.from_numpy(T), persistent=False)
        self.register_buffer("G", torch.from_numpy(G), persistent=False)
        self.k = round(FLOOR_HZ / (SR / n_fft)) | 1

    def forward(self, power):
        with torch.autocast(power.device.type, enabled=False):
            logS = 10 * torch.log10(power.float().clamp(min=1e-10))
            b, _, t, nf = logS.shape
            floor = F.avg_pool1d(logS.reshape(b * t, 1, nf), self.k, 1, self.k // 2, count_include_pad=False)
            white = ((logS - floor.reshape(logS.shape)).clamp(0, 40) / 40) ** SOFT_P
            band = (white @ self.W).clamp(min=0) ** (1 / SOFT_P)
            comb = ((white @ self.T).clamp(min=0) ** (1 / SOFT_P)) @ self.G
        return torch.cat([band, comb], 1)


class ExtraInputConv(nn.Module):
    """A pretrained 1-channel conv plus a zero-initialised conv over extra channels, so the model starts out
    exactly as the pretrained one and learns how much to use the new channels."""

    def __init__(self, conv, n_extra):
        super().__init__()
        self.conv = conv
        self.extra = nn.Conv2d(n_extra, conv.out_channels, conv.kernel_size, padding=conv.padding, bias=False)
        nn.init.zeros_(self.extra.weight)

    def forward(self, x):
        return self.conv(x[:, :1]) + self.extra(x[:, 1:])


def make_frontend(frontend):
    """(Spectrogram, LogmelFilterBank) for "mel" (CNN14's stock, n_fft 1024) or "logf" (bent mel, n_fft 4096).

    Both are fixed transforms (freeze_parameters=True), never trained.
    """
    if frontend not in ("mel", "logf"):
        raise ValueError(frontend)
    n_fft = 4096 if frontend == "logf" else 1024
    spec = Spectrogram(n_fft=n_fft, hop_length=HOP, win_length=n_fft, window="hann",
                       center=True, pad_mode="reflect", freeze_parameters=True)
    logmel = LogmelFilterBank(sr=SR, n_fft=n_fft, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
                              ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)
    if frontend == "logf":
        logmel.melW.data = torch.from_numpy(bent_mel_filters(n_fft))
    return spec, logmel


class DroneCNN14(nn.Module):
    def __init__(self, frontend="logf", checkpoint=None, n_classes=N_CLASSES):
        super().__init__()
        base = Cnn14_DecisionLevelMax(SR, 1024, HOP, N_MELS, FMIN, FMAX, 527)
        if checkpoint:
            base.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
        if frontend in ("logf", "logf_comb"):
            base.spectrogram_extractor, base.logmel_extractor = make_frontend("logf")
            init_bn(base.bn0)                      # bands now cover different frequencies: relearn
            base.bn0.reset_running_stats()
        elif frontend != "mel":
            raise ValueError(frontend)
        self.comb = None
        if frontend == "logf_comb":
            self.comb = CombFeatures(base.logmel_extractor.melW.data.clone())
            self.bn_extra = nn.ModuleList([nn.BatchNorm2d(N_MELS) for _ in range(2)])
            for bn in self.bn_extra:
                init_bn(bn)
            base.conv_block1.conv1 = ExtraInputConv(base.conv_block1.conv1, 2)
        base.fc_audioset = nn.Linear(2048, n_classes)
        init_layer(base.fc_audioset)
        self.base, self.frontend = base, frontend

    def body_parameters(self):
        head = {id(p) for p in self.head_parameters()}
        return [p for p in self.parameters() if id(p) not in head]

    def head_parameters(self):
        b = self.base
        extra = [*self.bn_extra.parameters(), *b.conv_block1.conv1.extra.parameters()] if self.comb else []
        return [*b.bn0.parameters(), *b.fc1.parameters(), *b.fc_audioset.parameters(), *extra]

    def features(self, wave):
        """(B, C, T, 64) normalised input: the log-f bands, plus the two comb channels for "logf_comb"."""
        b = self.base
        power = b.spectrogram_extractor(wave)
        x = b.logmel_extractor(power)                            # (B, 1, T, 64)
        x = b.bn0(x.transpose(1, 3)).transpose(1, 3)
        if self.comb is None:
            return x
        e = self.comb(power)
        e = [bn(e[:, i:i + 1].transpose(1, 3)).transpose(1, 3) for i, bn in enumerate(self.bn_extra)]
        return torch.cat([x, *e], 1)

    def forward(self, wave):
        """wave (B, n) at 32 kHz -> frame logits (B, T, n_classes) at 100 frames/s."""
        b = self.base
        x = self.features(wave)
        frames = x.shape[2]
        if self.training:
            x = b.spec_augmenter(x)
        for i, block in enumerate([b.conv_block1, b.conv_block2, b.conv_block3,
                                   b.conv_block4, b.conv_block5, b.conv_block6]):
            x = block(x, pool_size=(2, 2) if i < 5 else (1, 1), pool_type="avg")
            x = F.dropout(x, p=0.2, training=self.training)
        x = x.mean(dim=3)
        x = F.max_pool1d(x, 3, 1, 1) + F.avg_pool1d(x, 3, 1, 1)
        x = F.dropout(x, p=0.5, training=self.training).transpose(1, 2)
        x = F.dropout(F.relu_(b.fc1(x)), p=0.5, training=self.training)
        logits = b.fc_audioset(x)                                  # (B, T/32, C)
        return pad_framewise_output(interpolate(logits, 32), frames)


class SmallCNN(nn.Module):
    """From-scratch baseline (~1.2M params): same log-frequency input and frame output as DroneCNN14."""

    def __init__(self, frontend="logf", n_classes=N_CLASSES, widths=(32, 64, 128, 256)):
        super().__init__()
        self.spectrogram_extractor, self.logmel_extractor = make_frontend(frontend)
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)
        self.bn0 = nn.BatchNorm2d(N_MELS)
        layers, c_in = [], 1
        for c in widths:
            layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                       nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                       nn.AvgPool2d((2, 2))]
            c_in = c
        self.convs = nn.Sequential(*layers)
        self.gru = nn.GRU(c_in, 128, batch_first=True, bidirectional=True)
        self.head = nn.Linear(256, n_classes)
        self.frontend = frontend

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if "extractor" not in n]

    def body_parameters(self):
        return []

    def forward(self, wave):
        x = self.logmel_extractor(self.spectrogram_extractor(wave))     # (B, 1, T, 64)
        frames = x.shape[2]
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        if self.training:
            x = self.spec_augmenter(x)
        x = self.convs(x).mean(dim=3).transpose(1, 2)                  # (B, T/16, C)
        x, _ = self.gru(x)
        logits = self.head(F.dropout(x, 0.3, self.training))
        return pad_framewise_output(interpolate(logits, 16), frames)
