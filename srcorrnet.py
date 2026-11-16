"""Adapt the official mono SR-CorrNet-L model to the café runner."""

from pathlib import Path

import torch
from torch.nn import functional as F

MODEL_ID = "shinuh/sr-corrnet-ss-1ch-wsj-fix-2spk-l-dm"
MODEL_REVISION = "364ea2e5db9b029af28b8aa8b8769371d760641f"
SAMPLE_RATE = 8000


class _SRCorrNet(torch.nn.Module):
    def __init__(self, separator):
        super().__init__()
        self.separator = separator
        self.frame_length = separator.stft.N
        self.frame_shift = separator.stft.stride

    def forward(self, mixture):
        if mixture.ndim != 3 or mixture.shape[:2] != (1, 1) or not mixture.shape[-1]:
            raise ValueError("SR-CorrNet-L takes one nonempty mono mixture at a time")

        sample_count = mixture.shape[-1]
        # The upstream STFT has no padding. Cover both borders and keep the
        # final partial hop so the output stays aligned with the input.
        border = self.frame_length - self.frame_shift
        tail = (-(sample_count + 2 * border - self.frame_length)) % self.frame_shift
        waveform = F.pad(mixture[0], (border, border + tail))
        result = self.separator.process_waveform(
            waveform, n_spks=torch.tensor(2, device=self.separator.device)
        )
        tracks = torch.stack(result["waveforms"])
        if tracks.shape != (2, waveform.shape[-1]):
            raise RuntimeError("SR-CorrNet-L returned unexpected waveform dimensions")

        # process_waveform divides the input by its standard deviation.
        tracks = tracks[:, border : border + sample_count] * (waveform.std() + 1e-8)
        return tracks.unsqueeze(0)


def load_srcorrnet(weights: Path | None, device: str) -> torch.nn.Module:
    """Load the public checkpoint or a local model.pt with adjacent config.yaml."""
    from sr_corrnet import SSInference

    if weights is None:
        from huggingface_hub import hf_hub_download

        config = hf_hub_download(MODEL_ID, "config.yaml", revision=MODEL_REVISION)
        checkpoint = hf_hub_download(MODEL_ID, "model.pt", revision=MODEL_REVISION)
    else:
        checkpoint = weights / "model.pt" if weights.is_dir() else weights
        config = checkpoint.with_name("config.yaml")
        if not checkpoint.is_file() or not config.is_file():
            raise FileNotFoundError("SR-CorrNet-L needs a model.pt file and its adjacent config.yaml")
    separator = SSInference.from_pretrained(config=config, checkpoint_path=checkpoint, device=device)
    if separator.engine.fs != SAMPLE_RATE:
        raise ValueError(f"Expected an {SAMPLE_RATE} Hz SR-CorrNet-L checkpoint")
    return _SRCorrNet(separator).eval()
