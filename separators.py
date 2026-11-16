"""Run speech separators at their native rates on a common experiment timeline."""

from dataclasses import dataclass

import numpy as np

from run_pilot import infer_waveforms, load_separator as load_convtasnet
from run_pilot import resample_audio, tracks_to_pcm


SAMPLE_RATE = 16000
DEFAULT_MODEL = "mossformer2"
MODEL_SPECS = {
    "mossformer2": ("alibabasglab/MossFormer2_SS_16K", 16000, 2),
    "srcorrnet": ("shinuh/sr-corrnet-ss-1ch-wsj-fix-2spk-l-dm", 8000, 2),
    "convtasnet": ("torchaudio/CONVTASNET_BASE_LIBRI2MIX", 8000, 2),
    "mossformer2-3spk": ("alibabasglab/mossformer2-wsj0mix-3spk", 8000, 3),
    "srcorrnet-3spk": ("shinuh/sr-corrnet-ss-1ch-wsj-var-2-3spk", 8000, 3),
}


@dataclass
class Separator:
    model: object
    name: str
    native_rate: int
    device: str
    num_speakers: int = 2

    def separate(self, mixture_pcm):
        waveform = mixture_pcm.astype(np.float32) / 32768
        native = resample_audio(waveform, SAMPLE_RATE, self.native_rate)
        tracks = infer_waveforms(self.model, native, self.device, num_speakers=self.num_speakers)
        aligned = resample_audio(tracks, self.native_rate, SAMPLE_RATE)
        if aligned.shape[-1] < mixture_pcm.size:
            raise RuntimeError("Resampling returned a shorter timeline than the input")
        return tracks_to_pcm(aligned[:, :mixture_pcm.size])


def load_separator(name, weights, device):
    model_id, native_rate, num_speakers = MODEL_SPECS[name]
    if name == "mossformer2":
        from mossformer import load_mossformer

        model = load_mossformer(weights, device)
    elif name == "srcorrnet":
        from srcorrnet import load_srcorrnet

        model = load_srcorrnet(weights, device)
    elif name == "mossformer2-3spk":
        from three_speaker_backend import load_mossformer2_3spk

        model = load_mossformer2_3spk(weights, device)
    elif name == "srcorrnet-3spk":
        from three_speaker_backend import load_srcorrnet_3spk

        model = load_srcorrnet_3spk(weights, device)
    else:
        model = load_convtasnet(weights).to(device)
    return Separator(model, model_id, native_rate, device, num_speakers)
