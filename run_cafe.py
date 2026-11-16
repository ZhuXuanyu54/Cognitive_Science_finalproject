"""Run controlled café speech separation, or separate an existing mono WAV."""

from argparse import ArgumentParser
import csv
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.io import wavfile

from evaluate import evaluate
from run_pilot import read_mono
from separators import DEFAULT_MODEL, MODEL_SPECS, SAMPLE_RATE, load_separator

TRIPLE_MODEL = "mossformer2-3spk"
TRIPLE_CONDITIONS = (
    (0.50, 0, "50% 共同重叠 · 等电平"),
    (1.00, 0, "100% 共同重叠 · 等电平"),
    (1.00, -6, "100% 共同重叠 · 第三路 −6 dB"),
)


def write_audio(root, case_id, mixture, tracks, references=None):
    folder = root / case_id
    folder.mkdir()
    wavfile.write(folder / "mixture.wav", SAMPLE_RATE, mixture)
    track_paths = []
    for index, track in enumerate(tracks, start=1):
        filename = f"separated_{index}.wav"
        wavfile.write(folder / filename, SAMPLE_RATE, track)
        track_paths.append(f"{case_id}/{filename}")
    files = {"mixture": f"{case_id}/mixture.wav", "tracks": track_paths}
    if references is not None:
        reference_paths = []
        for index, track in enumerate(references, start=1):
            filename = f"reference_{index}.wav"
            wavfile.write(folder / filename, SAMPLE_RATE, track)
            reference_paths.append(f"{case_id}/{filename}")
        files["references"] = reference_paths
    return files


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
        target = read_mono(Path(target_file), SAMPLE_RATE)
        interferer = read_mono(Path(interferer_file), SAMPLE_RATE)
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
            tracks = model.separate(mixture)
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


def make_triple_mixture(clips, overlap_fraction, third_db):
    length = clips.shape[1]
    common_length = round(length * overlap_fraction)
    span = length - common_length
    offsets = (0, span // 2, span)
    levels = (0, 0, third_db)
    refs = np.zeros((3, length + span))
    rms_levels = [rms(clip) for clip in clips]
    if min(rms_levels) < 1e-6:
        raise ValueError("Each cropped source must contain audible speech")
    for i, (clip, offset, db) in enumerate(zip(clips, offsets, levels, strict=True)):
        refs[i, offset:offset + length] = clip * (rms_levels[0] / rms_levels[i]) * 10 ** (db / 20)
    peak = max(np.abs(refs).max(), np.abs(refs.sum(axis=0)).max())
    refs_pcm = np.rint(refs * (32768 * min(1.0, 0.9 / peak))).astype(np.int16)
    mixture_pcm = refs_pcm.sum(axis=0, dtype=np.int32).astype(np.int16)
    return mixture_pcm, refs_pcm, slice(span, length), offsets, levels


def run_triple_sweep(model, root, triples, clip_seconds):
    cases = []
    for group, (label, *files) in enumerate(triples, 1):
        sources = [read_mono(Path(path), SAMPLE_RATE) for path in files]
        shortest = min(source.size for source in sources)
        length = round(min(clip_seconds, shortest / SAMPLE_RATE) * SAMPLE_RATE)
        if length < 2 * SAMPLE_RATE:
            raise ValueError(f"{label}: use three WAV clips of at least 2 seconds")
        clips = np.stack([source[:length] for source in sources])
        clips -= clips.mean(axis=1, keepdims=True)
        for index, (fraction, third_db, condition) in enumerate(TRIPLE_CONDITIONS, 1):
            case_id = f"triple-{group:02d}-case-{index:02d}"
            mixture, refs, window, offsets, levels = make_triple_mixture(clips, fraction, third_db)
            shared_refs = refs[:, window].astype(np.float64)
            if np.min(np.std(shared_refs, axis=1)) < 1:
                raise ValueError(f"{label} / {condition}: each source needs speech in the overlap window")
            tracks = model.separate(mixture)
            result = evaluate(mixture[window], refs[:, window], tracks[:, window])
            tracks = tracks[result["reference_to_estimate"]]
            cases.append({
                "id": case_id,
                "label": condition,
                "pair_label": label,
                "sweep": "three-speaker",
                "num_speakers": 3,
                "source_names": [Path(path).name for path in files],
                "source_levels_db": list(levels),
                "source_start_seconds": [offset / SAMPLE_RATE for offset in offsets],
                "source_clip_seconds": length / SAMPLE_RATE,
                "overlap_fraction_of_shorter_clip": fraction,
                "overlap_start_seconds": window.start / SAMPLE_RATE,
                "overlap_seconds": (window.stop - window.start) / SAMPLE_RATE,
                "duration_seconds": mixture.size / SAMPLE_RATE,
                "files": write_audio(root, case_id, mixture, tracks, refs),
                "scores": {
                    "mean_sisdri": result["mean_sisdri"],
                    "min_sisdri": result["min_sisdri"],
                    **{f"source_{i}_sisdri": item["sisdri"]
                       for i, item in enumerate(result["per_source"], 1)},
                },
                "evaluation": result,
            })
            print(f"{case_id} {condition}: {result['mean_sisdri']:+.2f} dB", flush=True)
    fields = ("id", "pair_label", "label", "overlap_fraction_of_shorter_clip", "overlap_seconds")
    with (root / "scores.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[*fields, "source_3_db", *cases[0]["scores"]])
        writer.writeheader()
        for case in cases:
            writer.writerow({**{field: case[field] for field in fields},
                             "source_3_db": case["source_levels_db"][2], **case["scores"]})
    return cases


def run_recording(model, root, path):
    recording = read_mono(path, SAMPLE_RATE)
    gain = min(1.0, 0.9 / np.abs(recording).max())
    mixture = np.rint(recording * (gain * 32768)).astype(np.int16)
    case_id = "recording"
    return [{
        "id": case_id,
        "label": path.stem,
        "pair_label": "单麦克风录音",
        "sweep": "recording",
        "num_speakers": model.num_speakers,
        "duration_seconds": mixture.size / SAMPLE_RATE,
        "files": write_audio(root, case_id, mixture, model.separate(mixture)),
    }]


def main():
    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sweep = commands.add_parser("sweep", help="Compare level and overlap for one or more pairs")
    sweep.add_argument("--pair", nargs=3, action="append", required=True,
                       metavar=("LABEL", "TARGET.wav", "INTERFERER.wav"))
    sweep3 = commands.add_parser("sweep3", help="Evaluate three-way separation on controlled mixtures")
    sweep3.add_argument("--triple", nargs=4, action="append", required=True,
                        metavar=("LABEL", "FIRST.wav", "SECOND.wav", "THIRD.wav"))
    sweep3.add_argument("--clip-seconds", type=float, default=4,
                        help="Maximum per-source duration (default: 4 seconds)")
    recording = commands.add_parser("recording", help="Separate an existing mono WAV")
    recording.add_argument("wav", type=Path)
    selectable_models = (
        (sweep, 2, DEFAULT_MODEL), (sweep3, 3, TRIPLE_MODEL), (recording, None, DEFAULT_MODEL),
    )
    for command, count, default in selectable_models:
        command.add_argument("--output", type=Path, required=True, help="New output directory")
        command.add_argument("--model", default=default,
                             choices=[name for name, spec in MODEL_SPECS.items()
                                      if count is None or spec[2] == count],
                             help=f"Separator (default: {default})")
        command.add_argument("--weights", type=Path,
                             help="Local official checkpoint file or directory (see README)")
        command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Choose a new output directory: {args.output}")
    if args.command == "sweep3" and (not math.isfinite(args.clip_seconds) or args.clip_seconds < 2):
        parser.error("--clip-seconds must be finite and at least 2")

    import torch
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    try:
        model = load_separator(args.model, args.weights, args.device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".cafe-", dir=args.output.parent) as temporary:
            root = Path(temporary) / "results"
            root.mkdir()
            if args.command == "sweep":
                cases = run_sweep(model, root, args.pair)
            elif args.command == "sweep3":
                cases = run_triple_sweep(model, root, args.triple, args.clip_seconds)
            else:
                cases = run_recording(model, root, args.wav)
            manifest = {
                "model": model.name,
                "sample_rate_hz": SAMPLE_RATE,
                "model_sample_rate_hz": model.native_rate,
                "mixture_channels": 1,
                "num_speakers": model.num_speakers,
                "cases": cases,
            }
            (root / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            root.rename(args.output)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"{len(cases)} cases → {args.output}")


if __name__ == "__main__":
    main()
