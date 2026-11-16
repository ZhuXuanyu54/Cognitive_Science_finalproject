"""Bundle all evidence files into a single local offline HTML document."""

import argparse
import base64
import json
from pathlib import Path


def data_url(path, mime):
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build(media, output):
    template = Path(__file__).with_name("index.template.html").read_text(encoding="utf-8")
    data = json.loads((media/"results.json").read_text(encoding="utf-8"))
    data["mixture"] = data_url(media/"mixture.wav","audio/wav")
    for case in data["cases"]:
        ident = case["id"]
        case["image"] = data_url(media/f"{ident}-hist.png","image/png")
        case["tracks"] = [
            data_url(media/f"{ident}-{track}.wav","audio/wav")
            for track in (1,2)
        ] if len(case["peaks"])==2 else []
    output.write_text(template.replace("__FIXTURES__",json.dumps(data,ensure_ascii=False)),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("media",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    build(args.media,args.output)
