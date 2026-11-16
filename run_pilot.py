"""Run the initial two-speaker, single-channel separation experiment."""
from argparse import ArgumentParser
from math import gcd
import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from evaluate import evaluate

SAMPLE_RATE = 8000

def resample_audio(signal, input_rate, output_rate):
    if input_rate == output_rate:
        return signal
    factor = gcd(input_rate, output_rate)
    return resample_poly(signal, output_rate // factor, input_rate // factor, axis=-1)


def read_mono(path, sample_rate=SAMPLE_RATE):
    rate, x = wavfile.read(path)
    if x.ndim != 1 or not x.size:
        raise ValueError("Input must be a nonempty mono WAV")
    if x.dtype == np.uint8:
        x = (x.astype(np.float64) - 128) / 128
    elif np.issubdtype(x.dtype, np.signedinteger):
        x = x.astype(np.float64) / -float(np.iinfo(x.dtype).min)
    else:
        x = x.astype(np.float64)
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite audio samples: {path}")
    x = resample_audio(x, rate, sample_rate)
    x -= x.mean()
    if np.max(np.abs(x)) < 1e-6:
        raise ValueError("Input is silent")
    return x


def load_separator(weights=None):
    import torch
    import torchaudio

    if weights is None:
        return torchaudio.pipelines.CONVTASNET_BASE_LIBRI2MIX.get_model().eval()
    model = torchaudio.models.conv_tasnet_base(num_sources=2)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return model.eval()


def infer_waveforms(model, mixture, device="cpu", *, num_speakers=2):
    import torch

    tensor = torch.as_tensor(mixture, dtype=torch.float32, device=device)[None, None]
    with torch.inference_mode():
        estimates = model(tensor)[0].float().cpu().numpy()
    if estimates.shape != (num_speakers, mixture.size) or not np.isfinite(estimates).all():
        raise RuntimeError(
            f"Separator must return {num_speakers} finite tracks matching the input length"
        )
    return estimates


def tracks_to_pcm(estimates):
    peak = float(np.max(np.abs(estimates)))
    gain = min(1.0, 0.9 / peak) if peak > 0 else 1.0
    return np.rint(estimates * (gain * 32768)).astype(np.int16)


def separate_pcm(model, mix_pcm):
    estimates = infer_waveforms(model, mix_pcm.astype(np.float32) / 32768)
    return tracks_to_pcm(estimates)

def mix(first, second, offset_s):
    offset = round(offset_s * SAMPLE_RATE)
    length = max(len(first), offset + len(second))
    refs = np.zeros((2, length))
    refs[0, :len(first)] = first
    refs[1, offset:offset + len(second)] = second
    rms = np.array([np.sqrt(np.mean(first ** 2)), np.sqrt(np.mean(second ** 2))])
    refs *= (rms.min() / rms)[:, None]
    gain = min(1.0, .9 / max(np.max(np.abs(refs)), np.max(np.abs(refs.sum(axis=0)))))
    refs_pcm = np.rint(refs * gain * 32768).astype(np.int16)
    return refs_pcm.astype(np.int32).sum(0).astype(np.int16), refs_pcm, offset

def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("sources", type=Path, nargs=2)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--offset", type=float, default=1.2)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError("Choose a new output directory")
    mix_pcm, refs_pcm, offset = mix(*(read_mono(x) for x in args.sources), args.offset)
    estimates_pcm = separate_pcm(load_separator(), mix_pcm)
    result = {"model":"torchaudio/CONVTASNET_BASE_LIBRI2MIX", "sample_rate_hz":SAMPLE_RATE, "mixture_channels":1, "speaker_count":2, "offset_s":[0, offset/SAMPLE_RATE], "scores":evaluate(mix_pcm/32768, refs_pcm/32768, estimates_pcm/32768)}
    args.output.mkdir(parents=True)
    wavfile.write(args.output / "mixture.wav", SAMPLE_RATE, mix_pcm)
    for i, j in enumerate(result["scores"]["reference_to_estimate"]):
        wavfile.write(args.output / f"reference_{i+1}.wav", SAMPLE_RATE, refs_pcm[i])
        wavfile.write(args.output / f"separated_{i+1}.wav", SAMPLE_RATE, estimates_pcm[j])
    (args.output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
