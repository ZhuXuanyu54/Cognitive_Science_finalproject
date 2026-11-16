"""Load the official mono, 8 kHz three-speaker separators."""

from hashlib import file_digest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

MOSSFORMER2_3SPK_ID = "alibabasglab/mossformer2-wsj0mix-3spk"
MOSSFORMER2_3SPK_REVISION = "692a242d7e83aa8fc63c6a4b18153335a465cc1c"
SRCORRNET_3SPK_ID = "shinuh/sr-corrnet-ss-1ch-wsj-var-2-3spk"
SRCORRNET_3SPK_REVISION = "e523bd393c18db36257c5095b64d57709fa9dafb"
NATIVE_SAMPLE_RATE = 8000

_MOSSFORMER_CHECKSUMS = {
    "encoder.ckpt": "7a220724794baed750e0766e5112d1763a3afabc2156e07942c1d3e2ab77740b",
    "decoder.ckpt": "afe83c79bf392e1e0bdba5c97951b4b8fca62f521ef495fce245611065fb5923",
    "masknet.ckpt": "64fcad30db301d24aba695232637d134ca56569a16f7db20cabcb789a22b869a",
}
_SRCORRNET_CHECKSUMS = {
    "config.yaml": "4721cc0bd7f3707db3c13aa02c865ae15c56e44b67c1b94b81933d2c21ca4bf1",
    "model.pt": "21ab0209b25a0f5820b6f392c4dcb71a2a177298b3d3d963cbba795aaf1f53c1",
}


def _official_files(directory, repo, revision, checksums):
    if directory is None:
        from huggingface_hub import hf_hub_download

        paths = {
            name: Path(hf_hub_download(repo, name, revision=revision))
            for name in checksums
        }
    else:
        root = Path(directory)
        if not root.is_dir():
            raise FileNotFoundError(f"{repo} expects a --weights directory: {root}")
        paths = {name: root / name for name in checksums}

    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint file: {path}")
        with path.open("rb") as handle:
            digest = file_digest(handle, "sha256").hexdigest()
        if digest != checksums[name]:
            raise ValueError(f"Checkpoint checksum mismatch for {path} ({repo}@{revision})")
    return paths


class _MossFormer2ThreeSpeaker(nn.Module):
    def __init__(self, separator, kernel_size):
        super().__init__()
        self.separator = separator
        self.kernel_size = kernel_size
        self.stride = kernel_size // 2

    def forward(self, mixture):
        if mixture.ndim != 3 or mixture.shape[1] != 1 or not mixture.shape[-1]:
            raise ValueError("MossFormer2 requires nonempty mono input shaped (batch, 1, samples)")
        sample_count = mixture.shape[-1]
        # Cover the last partial encoder hop; trim back to the input timeline.
        minimum = max(self.kernel_size, sample_count)
        padded = minimum + (-(minimum - self.kernel_size)) % self.stride
        waveform = F.pad(mixture[:, 0], (0, padded - sample_count))
        # The upstream wrapper returns (batch, time, speakers).
        estimates = self.separator(waveform)
        return estimates[:, :sample_count, :].transpose(1, 2).contiguous()


def load_mossformer2_3spk(weights: Path | None, device: str) -> nn.Module:
    """Load the published WSJ0-3mix model and preserve the project tensor contract."""
    from vendor.mossformer2_standalone.model.mossformer2 import Mossformer2Wrapper
    from vendor.mossformer2_standalone.model.mossformer2_configs import mossformer2_wsj0mix_3spk

    paths = _official_files(
        weights, MOSSFORMER2_3SPK_ID, MOSSFORMER2_3SPK_REVISION, _MOSSFORMER_CHECKSUMS
    )
    config = mossformer2_wsj0mix_3spk
    model = Mossformer2Wrapper(config).to(device)
    for name in ("encoder", "decoder", "masknet"):
        tensors = torch.load(paths[f"{name}.ckpt"], map_location="cpu", weights_only=True)
        getattr(model, name).load_state_dict(tensors, strict=True)
    return _MossFormer2ThreeSpeaker(model, config["encoder_kernel_size"]).eval()


class _SRCorrNetThreeSpeaker(nn.Module):
    def __init__(self, separator):
        super().__init__()
        self.separator = separator
        self.frame_length = separator.stft.N
        self.frame_shift = separator.stft.stride

    def forward(self, mixture):
        if mixture.ndim != 3 or mixture.shape[:2] != (1, 1) or not mixture.shape[-1]:
            raise ValueError("SR-CorrNet requires one nonempty mono mixture at a time")
        samples = mixture.shape[-1]
        border = self.frame_length - self.frame_shift
        tail = (-(samples + 2 * border - self.frame_length)) % self.frame_shift
        waveform = F.pad(mixture[0], (border, border + tail))
        count = torch.tensor(3, device=self.separator.device)
        tracks = torch.stack(self.separator.process_waveform(waveform, n_spks=count)["waveforms"])
        if tracks.shape != (3, waveform.shape[-1]):
            raise RuntimeError("SR-CorrNet did not return three time-aligned tracks")
        # The upstream inference standardizes the input. Restore its scale.
        tracks = tracks[:, border : border + samples] * (waveform.std() + 1e-8)
        return tracks.unsqueeze(0)


def load_srcorrnet_3spk(weights: Path | None, device: str) -> nn.Module:
    """Load the published variable-count model with the same known count of three."""
    from sr_corrnet import SSInference

    paths = _official_files(
        weights, SRCORRNET_3SPK_ID, SRCORRNET_3SPK_REVISION, _SRCORRNET_CHECKSUMS
    )
    separator = SSInference.from_pretrained(
        config=paths["config.yaml"], checkpoint_path=paths["model.pt"], device=device
    )
    if separator.engine.fs != NATIVE_SAMPLE_RATE:
        raise ValueError("Expected the 8 kHz SR-CorrNet checkpoint")
    return _SRCorrNetThreeSpeaker(separator).eval()
