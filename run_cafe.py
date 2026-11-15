"""Run controlled café speech separation, or separate an existing mono WAV."""

from argparse import ArgumentParser
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.io import wavfile

from evaluate import evaluate
from run_pilot import SAMPLE_RATE, load_separator, read_mono, separate_pcm


def write_audio(root, case_id, mixture, tracks, references=None):
    folder = root / case_id
    folder.mkdir()
    wavfile.write(folder / "mixture.wav", SAMPLE_RATE, mixture)
    track_paths = []
    for index, track in enumerate(tracks, start=1):
        filename = f"separated_{index}.wav"
        wavfile.write(folder / filename, SAMPLE_RATE, track)
        track_paths.append(f"{case_id}/{filename}")
    if references is not None:
        for index, track in enumerate(references, start=1):
            wavfile.write(folder / f"reference_{index}.wav", SAMPLE_RATE, track)
    return {"mixture": f"{case_id}/mixture.wav", "tracks": track_paths}


def rms(signal):
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))


def make_mixture(target, interferer, overlap_samples, interferer_db):
    offset = target.size - overlap_samples
    refs = np.zeros((2, offset + interferer.size))
    refs[0, :target.size] = target
    refs[1, offset:] = interferer * (rms(target) / rms(interferer)) * 10 ** (interferer_db / 20)
    peak = max(np.abs(refs).max(), np.abs(refs.sum(axis=0)).max())
    gain = min(1.0, 0.9 / peak)
    refs_pcm = np.rint(refs * (gain * 32768)).astype(np.int16)
    mixture_pcm = refs_pcm.sum(axis=0, dtype=np.int32).astype(np.int16)
    return mixture_pcm, refs_pcm, slice(offset, target.size)


def run_sweep(model, root, pairs):
    cases = []
    for pair_index, (name, target_file, interferer_file) in enumerate(pairs, start=1):
        target = read_mono(Path(target_file))
        interferer = read_mono(Path(interferer_file))
        shorter = min(target.size, interferer.size)
        if shorter < SAMPLE_RATE * 2:
            raise ValueError(f"{name}: use two WAV clips of at least 2 seconds")
        conditions = [
            ("level", -6, 0.50, "干扰声低 6 dB"),
            ("level", 0, 0.50, "两路等电平"),
            ("level", 6, 0.50, "干扰声高 6 dB"),
            ("overlap", 0, 0.25, "短重叠"),
            ("overlap", 0, 0.90, "长重叠"),
        ]
        for case_index, (sweep, db, fraction, label) in enumerate(conditions, start=1):
            case_id = f"pair-{pair_index:02d}-case-{case_index:02d}"
            count = round(shorter * fraction)
            mixture, refs, interval = make_mixture(target, interferer, count, db)
            overlap_levels = [rms(ref[interval].astype(np.float64) - ref[interval].mean()) for ref in refs]
            if min(overlap_levels) < 1:
                raise ValueError(f"{name} / {label}: each source needs speech in the overlap window")
            tracks = separate_pcm(model, mixture)
            # Use only the shared clip interval for scoring and channel assignment.
            result = evaluate(mixture[interval], refs[:, interval], tracks[:, interval])
            tracks = tracks[result["reference_to_estimate"]]
            target_score, interferer_score = result["per_source"]
            cases.append({
                "id": case_id,
                "label": label,
                "pair_label": name,
                "sweep": sweep,
                "source_names": [Path(target_file).name, Path(interferer_file).name],
                "interferer_db": db,
                "overlap_interferer_db": 20 * float(np.log10(overlap_levels[1] / overlap_levels[0])),
                "overlap_fraction_of_shorter_clip": fraction,
                "overlap_start_seconds": interval.start / SAMPLE_RATE,
                "overlap_seconds": count / SAMPLE_RATE,
                "duration_seconds": mixture.size / SAMPLE_RATE,
                "files": write_audio(root, case_id, mixture, tracks, refs),
                "scores": {
                    "target_sisdr": target_score["sisdr"],
                    "target_sisdri": target_score["sisdri"],
                    "interferer_sisdr": interferer_score["sisdr"],
                    "interferer_sisdri": interferer_score["sisdri"],
                    "mean_sisdri": result["mean_sisdri"],
                },
                "evaluation": result,
            })
    write_scores(root, cases)
    return cases


def write_scores(root, cases):
    fields = [
        "id", "pair_label", "label", "sweep", "interferer_db",
        "overlap_fraction_of_shorter_clip", "overlap_interferer_db", "overlap_seconds",
    ]
    with (root / "scores.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields + list(cases[0]["scores"]))
        writer.writeheader()
        for case in cases:
            writer.writerow({**{field: case[field] for field in fields}, **case["scores"]})


def run_recording(model, root, path):
    recording = read_mono(path)
    gain = min(1.0, 0.9 / np.abs(recording).max())
    mixture = np.rint(recording * (gain * 32768)).astype(np.int16)
    case_id = "recording"
    return [{
        "id": case_id,
        "label": path.stem,
        "pair_label": "单麦克风录音",
        "sweep": "recording",
        "duration_seconds": mixture.size / SAMPLE_RATE,
        "files": write_audio(root, case_id, mixture, separate_pcm(model, mixture)),
    }]


def main():
    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sweep = commands.add_parser("sweep", help="Compare level and overlap for one or more pairs")
    sweep.add_argument("--pair", nargs=3, action="append", required=True,
                       metavar=("LABEL", "TARGET.wav", "INTERFERER.wav"))
    recording = commands.add_parser("recording", help="Separate an existing two-voice mono WAV")
    recording.add_argument("wav", type=Path)
    for command in (sweep, recording):
        command.add_argument("--output", type=Path, required=True, help="New output directory")
        command.add_argument("--weights", type=Path, help="Local official TorchAudio checkpoint")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Choose a new output directory: {args.output}")

    import torch
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = load_separator(args.weights)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with TemporaryDirectory(prefix=".cafe-", dir=args.output.parent) as temporary:
            root = Path(temporary) / "results"
            root.mkdir()
            if args.command == "sweep":
                cases = run_sweep(model, root, args.pair)
            else:
                cases = run_recording(model, root, args.wav)
            manifest = {
                "model": "torchaudio/CONVTASNET_BASE_LIBRI2MIX",
                "sample_rate_hz": SAMPLE_RATE,
                "mixture_channels": 1,
                "cases": cases,
            }
            (root / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            root.rename(args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"{len(cases)} cases → {args.output}")


if __name__ == "__main__":
    main()
