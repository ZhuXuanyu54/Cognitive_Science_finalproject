"""Compare the overlap scores from multiple runs of the same café sweep."""

import csv
import hashlib
import json
from argparse import ArgumentParser
from pathlib import Path
from statistics import mean


def checksum(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read_run(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    signatures, rows = [], []
    for case in manifest["cases"]:
        if "scores" not in case:
            raise ValueError(f"{root} is not a scored sweep")
        mixture = root / case["files"]["mixture"]
        count = len(case["files"]["tracks"])
        references = case["files"].get("references")
        if references is None:
            ref_paths = [mixture.with_name(f"reference_{i}.wav") for i in range(1, count + 1)]
        else:
            ref_paths = [root / name for name in references]
        if len(ref_paths) != count:
            raise ValueError(f"{case['id']} needs one reference for each scored track")
        source_files = [mixture, *ref_paths]
        signatures.append((case["id"], case["pair_label"], case["overlap_start_seconds"],
                           case["overlap_seconds"], tuple(checksum(path) for path in source_files)))
        rows.append({
            "model": manifest["model"],
            "pair": case["pair_label"],
            "case": case["id"],
            "interferer_db": case.get("interferer_db"),
            "overlap_fraction": case["overlap_fraction_of_shorter_clip"],
            "num_speakers": count,
            "source_3_db": case["source_levels_db"][2] if count == 3 else None,
            "mean_sisdr_db": case["evaluation"]["mean_sisdr"],
            "mean_sisdri_db": case["evaluation"]["mean_sisdri"],
            "min_sisdri_db": case["evaluation"]["min_sisdri"],
        })
    if not rows:
        raise ValueError(f"{root} has no scored cases")
    return rows, (manifest["sample_rate_hz"], signatures)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="Sweep output directories")
    parser.add_argument("--csv", type=Path, help="Write one row per case and model")
    args = parser.parse_args()
    if len(args.runs) < 2:
        parser.error("Provide at least two sweep output directories")
    try:
        runs = [read_run(root) for root in args.runs]
        if any(signature != runs[0][1] for _, signature in runs):
            raise ValueError("Runs must contain the same mixes, references, and overlap windows in the same order")
        ranking = sorted((rows for rows, _ in runs),
                         key=lambda rows: mean(row["mean_sisdri_db"] for row in rows), reverse=True)
        print(f"{'Mean SI-SDRi':>13}  Model ({len(ranking[0])} shared cases)")
        for rows in ranking:
            print(f"{mean(row['mean_sisdri_db'] for row in rows):>10.2f} dB  {rows[0]['model']}")
        if args.csv:
            with args.csv.open("x", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=ranking[0][0].keys())
                writer.writeheader()
                for rows in ranking:
                    writer.writerows(rows)
    except (KeyError, OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
