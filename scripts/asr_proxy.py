"""Optional multilingual Whisper transcription, explicitly an intelligibility proxy.

Run before/after acoustic training. No MOS or naturalness claim is made.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from faster_whisper import WhisperModel
from faster_whisper.utils import download_model
from shared.io import ROOT, write_json, read_jsonl

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--references", action="store_true")
parser.add_argument("--model", choices=["small", "large-v3"], default="large-v3")
parser.add_argument("--manifest", default="data/test_sentences.json")
parser.add_argument("--output", help="Optional output JSON path")
args = parser.parse_args()
revisions = {
    "small": "536b0662742c02347bc0e980a01041f333bce120",
    "large-v3": "edaa852ec7e145841d8ffdb056a99866b5f0a478",
}
model_path = download_model(
    "Systran/faster-whisper-" + args.model,
    cache_dir=str(ROOT / "artifacts/asr"),
    revision=revisions[args.model],
)
model = WhisperModel(model_path, device="cuda", compute_type="float16")
items = []
if args.references:
    for entry in json.loads((ROOT / "analysis/vocoder_validation.json").read_text()):
        path = (
            ROOT
            / "artifacts/vocoder_validation"
            / f'{entry["language"]}_{entry["speaker_id"]}_{entry["source_rate"]}.wav'
        )
        items.append(
            dict(
                path=str(path),
                language=entry["language"],
                source_id=entry["source_id"],
                source_kind="vocoder_reconstruction",
            )
        )
        rows = read_jsonl(ROOT / "data/manifests" / f'{entry["language"]}_val.jsonl')
        row = next(row for row in rows if row["source_id"] == entry["source_id"])
        items.append(
            dict(
                path=str(ROOT / row["audio_path"]),
                language=entry["language"],
                source_id=entry["source_id"],
                source_kind="original_reference",
            )
        )
else:
    for entry in json.loads(Path(args.manifest).read_text()):
        path = ROOT / entry["output_audio_filename"]
        if path.exists():
            items.append(
                dict(
                    path=str(path),
                    language=entry["language"],
                    source_text=entry["original_text"],
                    test_id=entry["test_id"],
                )
            )
results = []
for item in items:
    segments, info = model.transcribe(
        item["path"],
        language="en" if item["language"] == "english" else "te",
        beam_size=5,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    result = dict(
        item,
        asr_text=" ".join(s.text.strip() for s in segments),
        asr_model=args.model,
        asr_revision=revisions[args.model],
        note="Automatic intelligibility proxy; no human listening assessment or naturalness score",
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    results.append(result)
write_json(
    (
        Path(args.output)
        if args.output
        else ROOT
        / "analysis"
        / ("vocoder_asr_proxy.json" if args.references else "asr_proxy.json")
    ),
    results,
)
