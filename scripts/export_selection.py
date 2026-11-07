"""Publish only trace IDs/durations/hashes, never dataset audio or transcripts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.io import ROOT, read_jsonl, write_json, sha256

records = {}
for language in ["english", "telugu"]:
    for split in ["train", "val"]:
        path = ROOT / "data/manifests" / f"{language}_{split}.jsonl"
        if not path.exists():
            continue
        rows = read_jsonl(path)
        records[f"{language}_{split}"] = dict(
            manifest_sha256=sha256(path),
            rows=[
                {
                    k: r.get(k)
                    for k in [
                        "source_id",
                        "source_recording_id",
                        "speaker_id",
                        "duration_seconds",
                        "audio_sha256",
                    ]
                }
                for r in rows
            ],
        )
write_json(ROOT / "data/manifests/selection_index.json", records)
