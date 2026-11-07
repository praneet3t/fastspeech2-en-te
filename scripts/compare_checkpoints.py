"""Compare checkpoint intelligibility on a fixed validation-only probe.

WER is a Whisper transcription proxy, not a naturalness or human listening score.
Generated probe audio and corpus text stay under ignored artifacts/.
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.io import ROOT, load_config, read_jsonl, write_json
from shared.inference import synthesize
from shared.text import normalize
import re
import regex


def words(text, language="english"):
    normalized, _ = normalize(text, language)
    if language == "telugu":
        return regex.findall(r"[\p{L}\p{M}]+", normalized)
    return re.findall(r"[a-z]+(?:'[a-z]+)*", normalized)


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(hypothesis, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b))
            )
        previous = current
    return previous[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utterances", type=int, default=6)
    parser.add_argument("--language", choices=["english", "telugu"], default="english")
    args = parser.parse_args()
    language = args.language
    name = language.title()
    config = load_config(ROOT / name / "config.yaml")
    if not (ROOT / "analysis" / language / "training_complete.json").exists():
        raise ValueError(
            "Wait for training to finish before final checkpoint comparison"
        )
    eligible = [
        r
        for r in read_jsonl(ROOT / "data/features" / language / "val.jsonl")
        if 5 <= r["duration_seconds"] <= 12 and len(words(r["raw_text"], language)) >= 8
    ]
    random.Random(2026).shuffle(eligible)
    rows = eligible[: args.utterances]
    if len(rows) != args.utterances:
        raise ValueError("Insufficient validation probe recordings")
    base = ROOT / "artifacts" / f"{language}_checkpoint_probe"
    base.mkdir(parents=True, exist_ok=True)
    combined = []
    for role in ["best", "latest"]:
        manifest = base / f"{role}.json"
        records = [
            dict(
                test_id=f"{role}_{i:02d}",
                language=language,
                original_text=row["raw_text"],
                source_id=row["source_id"],
                speaker_id=row["speaker_id"],
                reference_split="val",
            )
            for i, row in enumerate(rows)
        ]
        write_json(manifest, records)
        synthesize(
            config,
            manifest,
            ROOT / name / "checkpoints" / f"{role}.pt",
            output_root=base / role,
        )
        combined += json.loads(manifest.read_text())
    manifest = base / "combined.json"
    write_json(manifest, combined)
    output = base / "asr.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/asr_proxy.py"),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
        check=True,
    )
    transcriptions = {r["test_id"]: r for r in json.loads(output.read_text())}
    result = dict(
        seed=2026,
        utterances=len(rows),
        selection="5–12 s validation-only utterances with >=8 normalized words; seeded shuffle",
        asr="Whisper large-v3 edaa852ec7e145841d8ffdb056a99866b5f0a478",
        metric="word edit distance / normalized reference words; ASR intelligibility proxy only",
        candidates={},
    )
    for role in ["best", "latest"]:
        details = []
        for i, row in enumerate(rows):
            test_id = f"{role}_{i:02d}"
            text = transcriptions[test_id]
            prediction = json.loads(
                (base / role / language / "predictions" / f"{test_id}.json").read_text()
            )
            reference = words(row["raw_text"], language)
            hypothesis = words(text["asr_text"], language)
            details.append(
                dict(
                    source_id=row["source_id"],
                    speaker_id=row["speaker_id"],
                    reference_duration_seconds=row["duration_seconds"],
                    reference_words=len(reference),
                    word_errors=edit_distance(reference, hypothesis),
                )
            )
        total = sum(r["reference_words"] for r in details)
        errors = sum(r["word_errors"] for r in details)
        result["candidates"][role] = dict(
            step=prediction["checkpoint_step"],
            checkpoint_sha256=prediction["checkpoint_sha256"],
            reference_words=total,
            word_errors=errors,
            asr_wer=errors / total,
            utterances=details,
        )
    write_json(ROOT / "analysis" / language / "checkpoint_probe.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
