"""Run the initial two-speaker, single-channel separation experiment."""
from argparse import ArgumentParser
from math import gcd
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from evaluate import evaluate

SAMPLE_RATE = 8000

def read_mono(path):
    rate, x = wavfile.read(path)
    if x.ndim != 1 or not x.size:
        raise ValueError("Input must be a nonempty mono WAV")
    if np.issubdtype(x.dtype, np.signedinteger):
        x = x.astype(np.float64) / -float(np.iinfo(x.dtype).min)
    else:
        x = x.astype(np.float64)
    if rate != SAMPLE_RATE:
        factor = gcd(rate, SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // factor, rate // factor)
    x -= x.mean()
    if np.max(np.abs(x)) < 1e-6:
        raise ValueError("Input is silent")
    return x

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
    import torch, torchaudio
    model = torchaudio.pipelines.CONVTASNET_BASE_LIBRI2MIX.get_model().eval()
    with torch.inference_mode():
        estimates = model(torch.from_numpy(mix_pcm.astype(np.float32) / 32768)[None, None])[0].numpy()
    gain = min(1.0, .9 / np.max(np.abs(estimates)))
    estimates_pcm = np.rint(estimates * gain * 32768).astype(np.int16)
    result = {"model":"torchaudio/CONVTASNET_BASE_LIBRI2MIX", "sample_rate_hz":SAMPLE_RATE, "mixture_channels":1, "speaker_count":2, "offset_s":[0, offset/SAMPLE_RATE], "scores":evaluate(mix_pcm/32768, refs_pcm/32768, estimates_pcm/32768)}
    args.output.mkdir(parents=True)
    wavfile.write(args.output / "mixture.wav", SAMPLE_RATE, mix_pcm)
    for i, j in enumerate(result["scores"]["reference_to_estimate"]):
        wavfile.write(args.output / f"reference_{i+1}.wav", SAMPLE_RATE, refs_pcm[i])
        wavfile.write(args.output / f"separated_{i+1}.wav", SAMPLE_RATE, estimates_pcm[j])
    (args.output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
