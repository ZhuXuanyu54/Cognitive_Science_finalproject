"""Package a verified run as a standalone, offline listening page."""

import argparse
import base64
from html import escape
import io
import json
import math
import sys
import wave
from array import array
from pathlib import Path


HERE = Path(__file__).resolve().parent


def pcm16(path):
    with wave.open(str(path), "rb") as source:
        if source.getsampwidth() != 2 or source.getnchannels() != 1:
            raise ValueError(f"{path.name} must be a mono PCM16 WAV")
        frames = source.getnframes()
        rate = source.getframerate()
        samples = array("h", source.readframes(frames))
    if len(samples) != frames or not samples:
        raise ValueError(f"{path.name} is empty or incomplete")
    if sys.byteorder == "big":
        samples.byteswap()
    return rate, samples


def wav_data_url(samples, rate, gain):
    adjusted = array("h", (round(sample * gain) for sample in samples))
    if sys.byteorder == "big":
        adjusted.byteswap()
    target = io.BytesIO()
    with wave.open(target, "wb") as result:
        result.setnchannels(1)
        result.setsampwidth(2)
        result.setframerate(rate)
        result.writeframes(adjusted.tobytes())
    return "data:audio/wav;base64," + base64.b64encode(target.getvalue()).decode("ascii")


def bundle_case(source, item):
    files = item["files"]
    tracks = files["tracks"]
    if len(tracks) not in (2, 3):
        raise ValueError(f'{item["id"]} must list two or three separated tracks')
    decoded = [pcm16(source / name) for name in [files["mixture"], *tracks]]
    rate, first = decoded[0]
    if any(file_rate != rate or len(samples) != len(first) for file_rate, samples in decoded):
        raise ValueError(f'{item["id"]} must contain aligned recordings at one sample rate')

    loudness = [math.sqrt(sum(value * value for value in samples) / len(samples)) for _, samples in decoded]
    ratios = [max(map(abs, samples)) / rms for (_, samples), rms in zip(decoded, loudness) if rms > 0]
    target_rms = min(2400, 31000 / max(ratios)) if ratios else 2400
    urls = [
        wav_data_url(samples, rate, target_rms / rms if rms else 1)
        for (_, samples), rms in zip(decoded, loudness, strict=True)
    ]
    return {**item, "files": {"mixture": urls[0], "tracks": urls[1:]}}


def bundle(manifest_path, output, chosen_cases, credits_file=None):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["cases"]
    if chosen_cases:
        by_id = {item["id"]: item for item in entries}
        missing = sorted(set(chosen_cases) - by_id.keys())
        if missing:
            raise ValueError(f"Unknown cases: {', '.join(missing)}")
        entries = [by_id[case] for case in chosen_cases]
    if not entries:
        raise ValueError("The manifest must contain at least one case")
    cases = [bundle_case(manifest_path.parent, item) for item in entries]
    payload = json.dumps({"model": manifest.get("model"), "cases": cases}, ensure_ascii=False)
    payload = payload.replace("<", "\\u003c")

    html = (HERE / "index.html").read_text(encoding="utf-8")
    css = (HERE / "style.css").read_text(encoding="utf-8")
    if credits_file:
        css += "\n.demo-credits { margin-block-start: var(--sp-3); font-size: var(--small); overflow-wrap: anywhere; }\n"
    script = (HERE / "demo.js").read_text(encoding="utf-8")
    html = html.replace('  <link rel="stylesheet" href="style.css">\n  <script src="demo.js" defer></script>',
                        f"  <style>\n{css}\n  </style>")
    three_track_only = all(len(item["files"]["tracks"]) == 3 for item in cases)
    title = "单通道三人语音分离演示" if three_track_only else "单通道语音分离离线演示"
    html = html.replace("咖啡厅语音分离试听", title)
    html = html.replace(
        "在同一播放位置比较原始混音和分离结果，选择想听的主轨。",
        "音轨由模型预先分离。试听音量按 RMS 对齐，评分沿用原始结果。",
    )
    if credits_file:
        paragraphs = credits_file.read_text(encoding="utf-8").strip().splitlines()
        credits = "".join(f"<p>{escape(line)}</p>" for line in paragraphs if line.strip())
        html = html.replace("</main>", f'<details class="demo-credits"><summary>音频来源</summary>{credits}</details>\n</main>')
    html = html.replace(
        "</body>",
        f'<script id="embedded-manifest" type="application/json">{payload}</script>\n<script>\n{script}\n</script>\n</body>',
    )
    output.write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="run manifest.json")
    parser.add_argument("--output", type=Path, required=True, help="destination HTML")
    parser.add_argument("--case", action="append", help="retain a case ID, in the given order")
    parser.add_argument("--credits", type=Path, help="optional UTF-8 dataset and license attributions")
    args = parser.parse_args()
    bundle(args.manifest, args.output, args.case, args.credits)


if __name__ == "__main__":
    main()
