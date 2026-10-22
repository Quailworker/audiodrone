"""CNN14 (NeMo/PANNs Cnn14_DecisionLevelMax) with a swappable front end and a drone head.

Front ends (design.md 5.8.4):
  "mel"  - stock log-mel, n_fft 1024 (option A, baseline)
  "logf" - mel bent at 50 Hz (log-spaced over the whole drone range), n_fft 4096 (option B)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import LogmelFilterBank, Spectrogram

from .panns_models import Cnn14_DecisionLevelMax, init_bn, init_layer
from .panns_pytorch_utils import interpolate, pad_framewise_output

SR, HOP, N_MELS, FMIN, FMAX = 32000, 320, 64, 50.0, 14000.0
CLASSES = ["drone_electric", "drone_engine", "ground_engine"]


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


class DroneCNN14(nn.Module):
    def __init__(self, frontend="logf", checkpoint=None, n_classes=len(CLASSES)):
        super().__init__()
        base = Cnn14_DecisionLevelMax(SR, 1024, HOP, N_MELS, FMIN, FMAX, 527)
        if checkpoint:
            base.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
        if frontend == "logf":
            n_fft = 4096
            base.spectrogram_extractor = Spectrogram(n_fft=n_fft, hop_length=HOP, win_length=n_fft, window="hann",
                                                     center=True, pad_mode="reflect", freeze_parameters=True)
            base.logmel_extractor = LogmelFilterBank(sr=SR, n_fft=n_fft, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
                                                     ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)
            base.logmel_extractor.melW.data = torch.from_numpy(bent_mel_filters(n_fft))
            init_bn(base.bn0)                      # bands now cover different frequencies: relearn
            base.bn0.reset_running_stats()
        elif frontend != "mel":
            raise ValueError(frontend)
        base.fc_audioset = nn.Linear(2048, n_classes)
        init_layer(base.fc_audioset)
        self.base, self.frontend = base, frontend

    def body_parameters(self):
        head = {id(p) for p in self.head_parameters()}
        return [p for p in self.parameters() if id(p) not in head]

    def head_parameters(self):
        b = self.base
        return [*b.bn0.parameters(), *b.fc1.parameters(), *b.fc_audioset.parameters()]

    def features(self, wave):
        b = self.base
        x = b.logmel_extractor(b.spectrogram_extractor(wave))   # (B, 1, T, 64)
        return x

    def forward(self, wave):
        """wave (B, n) at 32 kHz -> frame logits (B, T, n_classes) at 100 frames/s."""
        b = self.base
        x = self.features(wave)
        frames = x.shape[2]
        x = b.bn0(x.transpose(1, 3)).transpose(1, 3)
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
