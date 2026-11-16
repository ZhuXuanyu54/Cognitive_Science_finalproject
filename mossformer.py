"""Load the official 16 kHz, two-speaker ClearVoice MossFormer2 model."""

from argparse import Namespace
from pathlib import Path
from urllib.error import URLError

import torch
from torch import nn
from torch.nn import functional as F


_REPOSITORY = "alibabasglab/MossFormer2_SS_16K"
_REVISION = "407cb030cd66340918ebb6c8cc63b18f8592cdbe"
_CHECKPOINT_URL = (
    f"https://huggingface.co/{_REPOSITORY}/resolve/{_REVISION}/last_best_checkpoint.pt"
)
_MODELSCOPE_URL = (
    "https://modelscope.cn/models/iic/ClearerVoice-Studio/resolve/master/"
    "MossFormer2_SS_16K/last_best_checkpoint.pt"
)
_CHECKPOINT_SHA256 = "00a3a48bda492db1e829b85dd443f8f43a43039a3e90f1a24962ea9caf14a11a"
_ENCODER_KERNEL = 16
_TARGET_RMS = 10 ** (-25 / 20)


class _MossFormerAdapter(nn.Module):
    def __init__(self, separator: nn.Module) -> None:
        super().__init__()
        self.separator = separator

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        samples = mixture.shape[-1]
        waveform = mixture.squeeze(1)
        # ClearVoice normalizes the RMS of samples with above-average power.
        power = waveform.square()
        mean_power = power.mean(dim=-1, keepdim=True)
        active = power > mean_power
        count = active.sum(dim=-1, keepdim=True)
        active_power = (power * active).sum(dim=-1, keepdim=True) / count.clamp_min(1)
        gain = _TARGET_RMS / torch.where(count > 0, active_power, mean_power).sqrt().clamp_min(1e-8)

        # Pad to a complete encoder frame so the model covers the last sample.
        stride = _ENCODER_KERNEL // 2
        padded = max(_ENCODER_KERNEL, ((samples + stride - 1) // stride) * stride)
        waveform = F.pad(waveform * gain, (0, padded - samples))
        # Keep each clip in a single pass so speaker order is shared over the clip.
        tracks = self.separator(waveform)
        return torch.stack(tracks, dim=1)[..., :samples] / gain.unsqueeze(1)


def _download_checkpoint() -> Path:
    cache = Path(__file__).resolve().parent / "checkpoints"
    checkpoint = cache / f"mossformer2-ss-16k-{_REVISION[:7]}.pt"
    if not checkpoint.exists():
        cache.mkdir(parents=True, exist_ok=True)
        try:
            torch.hub.download_url_to_file(_CHECKPOINT_URL, str(checkpoint), _CHECKPOINT_SHA256)
        except (URLError, TimeoutError):
            torch.hub.download_url_to_file(_MODELSCOPE_URL, str(checkpoint), _CHECKPOINT_SHA256)
    return checkpoint


def load_mossformer(weights: Path | None, device: str) -> nn.Module:
    """Return a model accepting (batch, 1, samples) and returning two tracks."""
    try:
        from clearvoice.models.mossformer2_ss.mossformer2 import MossFormer2_SS_16K
    except ImportError as error:
        raise RuntimeError(
            "MossFormer2 requires ClearVoice. Install the project requirements."
        ) from error

    checkpoint = weights if weights is not None else _download_checkpoint()
    args = Namespace(
        encoder_kernel_size=_ENCODER_KERNEL,
        encoder_embedding_dim=512,
        mossformer_sequence_dim=512,
        num_mossformer_layer=24,
        num_spks=2,
    )
    separator = MossFormer2_SS_16K(args).model
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("MossFormer2 checkpoint must contain a model state dictionary")
    state = state.get("model", state)
    if not isinstance(state, dict):
        raise ValueError("Invalid 'model' state in the MossFormer2 checkpoint")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    separator.load_state_dict(state, strict=True)
    return _MossFormerAdapter(separator).to(device).eval()
