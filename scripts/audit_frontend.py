"""Confirm persisted supervised phones use exactly the current inference front end."""

import sys
import json
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.io import ROOT, read_jsonl, write_json
from shared.text import frontend, normalize
from shared.data import transcript_key

report = {}
tests = json.loads((ROOT / "data/test_sentences.json").read_text())
overlap = []
for language in ["english", "telugu"]:
    changes = Counter()
    failures = []
    test_keys = {
        transcript_key(normalize(row["original_text"], language)[0]): row["test_id"]
        for row in tests
        if row["language"] == language
    }
    count = 0
    for split in ["train", "val"]:
        for row in read_jsonl(ROOT / "data/manifests" / f"{language}_{split}.jsonl"):
            text = frontend(row["raw_text"], language)
            count += 1
            matched = test_keys.get(transcript_key(text.normalized))
            if matched:
                overlap.append(
                    dict(test_id=matched, source_id=row["source_id"], split=split)
                )
            if (
                text.words != row["words"]
                or text.pronunciations != row["pronunciations"]
            ):
                failures.append(row["source_id"])
            for change in text.transformations:
                changes[change.split(":")[0]] += 1
    report[language] = dict(
        utterances=count,
        mismatched_frontend_rows=failures,
        normalization_counts=dict(changes),
    )
    if failures:
        raise ValueError(
            f"{language} current front end differs from supervised dictionary: {failures[:10]}"
        )
write_json(ROOT / "analysis/frontend_audit.json", report)
write_json(
    ROOT / "analysis/test_overlap_audit.json",
    dict(
        comparison="Normalized text with punctuation/spacing removed and Telugu marks retained",
        selected_corpus_matches=overlap,
        supplied_test_count=len(tests),
    ),
)
if overlap:
    raise ValueError(f"Supplied tests overlap selected corpus: {overlap}")
print(report)
