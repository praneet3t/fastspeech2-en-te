"""Audit original parquet audio and deterministically select whole utterances."""

import hashlib
import io
import json
import os
import random
import re
import regex
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from shared.io import ROOT, read_jsonl, write_json, write_jsonl
from shared.text import frontend


def transcript_key(text):
    """Exact normalized comparison retains Telugu combining marks."""
    import unicodedata

    return regex.sub(
        r"[\p{P}\p{Z}\p{C}]+", "", unicodedata.normalize("NFC", text).casefold()
    )


def split_group_key(text):
    """Actual experiment's conservative skeleton, never model input.

    Python non-word removal also drops Telugu vowel signs. This deliberately
    preserves the recorded split's additional grouping of distinct spellings;
    it does not remove marks from raw text or linguistic-unit sequences.
    """
    import unicodedata

    return re.sub(r"\W+", "", unicodedata.normalize("NFC", text).casefold())


def audit(language: str, config: dict):
    source = config["dataset"]
    folder = ROOT / "data/raw" / language
    files = sorted(folder.glob("data/*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No {language} parquet files. Run python scripts/download_data.py {language}"
        )
    rows = []
    known_hashes = set()
    for file in files:
        parquet = pq.ParquetFile(file)
        print(file.name, parquet.schema_arrow, flush=True)
        index = 0
        for batch in parquet.iter_batches(batch_size=16):
            for item in batch.to_pylist():
                source_id = f"{file.stem}:{index:06d}"
                index += 1
                row = dict(
                    source_id=source_id,
                    source_file=file.name,
                    source_row=index - 1,
                    dataset=source["repo"],
                    revision=source["revision"],
                    language=language,
                    raw_text=item.get("text", item.get("sentence", "")),
                    split="unused",
                    rejection_reason=None,
                )
                try:
                    audio = item.get("audio", item.get("audio_filepath"))
                    raw = audio["bytes"]
                    source_path = audio.get("path") or ""
                    chunk = re.fullmatch(
                        r"(\d+_f\d+)_chunk_\d+\.wav", Path(source_path).name
                    )
                    row["source_recording_id"] = (
                        chunk[1] if language == "english" and chunk else source_id
                    )
                    if raw is None:
                        raise ValueError(
                            "Parquet contains paths without embedded audio; source resolver required"
                        )
                    samples, rate = sf.read(
                        io.BytesIO(raw), dtype="float32", always_2d=True
                    )
                    if len(samples) == 0 or not np.isfinite(samples).all():
                        raise ValueError("Empty/non-finite audio")
                    mono = samples.mean(axis=1)
                    digest = hashlib.sha256(
                        mono.tobytes() + str(rate).encode()
                    ).hexdigest()
                    row.update(
                        original_audio_path=audio.get("path"),
                        sample_rate=rate,
                        channels=samples.shape[1],
                        sample_count=len(samples),
                        duration_seconds=len(samples) / rate,
                        audio_sha256=digest,
                        peak=float(np.max(np.abs(samples))),
                        clipped_fraction=float(np.mean(np.abs(samples) >= 0.999)),
                        quiet_fraction=float(np.mean(np.abs(mono) < 0.001)),
                        rms=float(np.sqrt(np.mean(mono.astype("float64") ** 2))),
                        normalization_status="original; no loudness normalization",
                        audio_integrity="valid",
                    )
                    if item.get("duration") is not None:
                        row["source_declared_duration_seconds"] = item["duration"]
                        if abs(item["duration"] - row["duration_seconds"]) > 0.02:
                            raise ValueError("Source duration disagrees with waveform")
                    if language == "telugu":
                        gender = item["gender"]
                        if gender not in (0, 1):
                            raise ValueError(f"Unexpected gender class: {gender}")
                        # Dataset card explicitly states one speaker in each gender class.
                        row["speaker_id"] = ["female", "male"][gender]
                        row["speaker_evidence"] = (
                            "HF gender class and dataset card: one speaker per class"
                        )
                    else:
                        # Use only an explicit source field; never parse arbitrary filenames.
                        speaker = next(
                            (
                                item[k]
                                for k in ("speaker_id", "speaker", "client_id")
                                if item.get(k) is not None
                            ),
                            None,
                        )
                        row["speaker_id"] = (
                            str(speaker) if speaker is not None else "unknown_mixture"
                        )
                        row["speaker_evidence"] = (
                            "explicit source field"
                            if speaker is not None
                            else "unrecoverable: unconditioned mixture limitation"
                        )
                    if digest in known_hashes:
                        row["rejection_reason"] = "duplicate_audio"
                    known_hashes.add(digest)
                    if not row["rejection_reason"]:
                        if row["rms"] < 0.0001:
                            row["rejection_reason"] = "near_silent"
                        elif row["clipped_fraction"] > 0.01:
                            row["rejection_reason"] = "excessive_clipping"
                        elif row["duration_seconds"] < 0.4:
                            row["rejection_reason"] = "too_short"
                    try:
                        text = frontend(row["raw_text"], language)
                        row.update(
                            normalized_text=text.normalized,
                            words=text.words,
                            pronunciations=text.pronunciations,
                            normalization_log=text.transformations,
                        )
                    except ValueError as error:
                        row["rejection_reason"] = "text_frontend: " + str(error)
                    # Store originals locally for repeatable alignment, never in Git.
                    out = folder / "audio" / f'{source_id.replace(":", "_")}.wav'
                    out.parent.mkdir(parents=True, exist_ok=True)
                    valid_existing = False
                    if out.exists():
                        try:
                            info = sf.info(out)
                            valid_existing = (
                                info.frames == len(samples)
                                and info.samplerate == rate
                                and info.channels == samples.shape[1]
                            )
                        except RuntimeError:
                            pass
                    if not valid_existing:
                        temporary = out.with_suffix(".tmp")
                        sf.write(
                            temporary, samples, rate, subtype="PCM_16", format="WAV"
                        )
                        os.replace(temporary, out)
                    row["audio_path"] = str(out.relative_to(ROOT))
                except (ValueError, KeyError, RuntimeError) as error:
                    row["rejection_reason"] = "invalid_source: " + str(error)
                    row["audio_integrity"] = "invalid"
                rows.append(row)
        print(f"Audited {len(rows)} rows", flush=True)
        write_jsonl(ROOT / "analysis" / f"{language}_audit.jsonl", rows)
    return rows


def select(
    rows: list[dict], seed: int, train_seconds=18000, val_seconds=3600, max_seconds=30
):
    """Group matching transcripts before split; greedily balance speaker hours.

    Audio duplicates are rejected by audit. Transcript matches across speakers
    remain usable, but always belong to one split. Durations are never truncated.
    """
    copied = [dict(row) for row in rows]
    eligible = []
    for row in copied:
        if row["rejection_reason"]:
            continue
        if row["duration_seconds"] > max_seconds:
            row["rejection_reason"] = f"longer_than_{max_seconds}s"
            continue
        eligible.append(row)
    # Connected components prevent both transcript and original recording
    # overlap, including when one transcript links two recordings.
    parent = list(range(len(eligible)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(eligible):
        for key in [
            ("text", split_group_key(row["raw_text"])),
            ("recording", row.get("source_recording_id", row["source_id"])),
        ]:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(eligible):
        groups[find(i)].append(row)
    candidates = list(groups.values())
    random.Random(seed).shuffle(candidates)
    available = Counter()
    for group in candidates:
        for row in group:
            available[row["speaker_id"]] += row["duration_seconds"]
    total = sum(available.values())
    if total < (train_seconds + val_seconds) * 0.95:
        raise ValueError(
            f"Only {total/3600:.3f} eligible hours with max_seconds={max_seconds}; inspect exclusions before relaxing limits"
        )
    targets = {
        split: {
            speaker: duration * hours / total for speaker, hours in available.items()
        }
        for split, duration in [("train", train_seconds), ("val", val_seconds)]
    }
    actual = {split: Counter() for split in targets}
    totals = Counter()
    # Assign whole transcript groups to the split with the larger relative deficit.
    for group in candidates:
        amount = Counter()
        for row in group:
            amount[row["speaker_id"]] += row["duration_seconds"]
        scores = {}
        for split, target in targets.items():
            if totals[split] >= sum(target.values()) or totals[split] + sum(
                amount.values()
            ) > 1.05 * sum(target.values()):
                continue
            scores[split] = sum(
                amount[s] * max(0, 1 - actual[split][s] / target[s]) for s in amount
            )
        if not scores:
            break
        split = max(scores, key=scores.get)
        for row in group:
            row["split"] = split
            actual[split][row["speaker_id"]] += row["duration_seconds"]
            totals[split] += row["duration_seconds"]
    for split, target in [("train", train_seconds), ("val", val_seconds)]:
        if abs(totals[split] - target) > 0.05 * target:
            raise ValueError(f"{split} duration {totals[split]} outside 5% target")
    train_speakers = {r["speaker_id"] for r in copied if r["split"] == "train"}
    if any(
        r["speaker_id"] not in train_speakers for r in copied if r["split"] == "val"
    ):
        raise ValueError("Validation contains unseen speakers")
    return copied


def summarize(rows, config):
    result = dict(
        dataset=config["dataset"],
        seed=config["seed"],
        max_seconds=config["selection"]["max_seconds"],
        strategy="seeded connected transcript/recording groups with duration-weighted speaker deficits; whole utterances",
        exclusions=dict(
            Counter(r["rejection_reason"] for r in rows if r["rejection_reason"])
        ),
        audited_utterances=len(rows),
        splits={},
    )
    for split in ["train", "val", "unused"]:
        selected = [
            r for r in rows if r["split"] == split and not r["rejection_reason"]
        ]
        duration = np.array([r["duration_seconds"] for r in selected])
        speakers = Counter()
        for row in selected:
            speakers[row["speaker_id"]] += row["duration_seconds"] / 3600
        result["splits"][split] = dict(
            utterances=len(selected),
            hours=float(duration.sum() / 3600),
            speaker_count=len(speakers) if config["language"] == "telugu" else None,
            conditioning_groups=len(speakers),
            hours_per_group=dict(speakers),
            hours_per_speaker=(
                dict(speakers) if config["language"] == "telugu" else None
            ),
            duration_percentiles_seconds=(
                np.percentile(duration, [0, 25, 50, 75, 95, 100]).tolist()
                if len(duration)
                else []
            ),
        )
    train, val = [
        [r for r in rows if r["split"] == split] for split in ["train", "val"]
    ]
    for key in ["source_id", "audio_sha256", "source_recording_id"]:
        if {r.get(key, r["source_id"]) for r in train} & {
            r.get(key, r["source_id"]) for r in val
        }:
            raise ValueError(f"Train/validation overlap: {key}")
    if {transcript_key(r["raw_text"]) for r in train} & {
        transcript_key(r["raw_text"]) for r in val
    }:
        raise ValueError("Exact normalized transcript overlap")
    result["audited_hours"] = sum(r.get("duration_seconds", 0) for r in rows) / 3600
    durations = [r["duration_seconds"] for r in rows if "duration_seconds" in r]
    result["raw_duration_percentiles_seconds"] = np.percentile(
        durations, [0, 25, 50, 75, 95, 99, 100]
    ).tolist()
    result["sample_rate_counts"] = dict(Counter(r.get("sample_rate") for r in rows))
    result["channel_counts"] = dict(Counter(r.get("channels") for r in rows))
    counts = Counter(transcript_key(r["raw_text"]) for r in rows)
    result["repeated_normalized_transcript_rows"] = sum(
        n - 1 for n in counts.values() if n > 1
    )
    result["speaker_identity_status"] = (
        "two explicit gender-labelled speakers, as documented by source"
        if config["language"] == "telugu"
        else "unrecoverable; unknown_mixture is a bookkeeping group, NOT one actual speaker"
    )
    result["known_speaker_count"] = 2 if config["language"] == "telugu" else None
    result["transcript_policy"] = (
        "Split grouping uses NFC/casefold and Python non-word removal, including Telugu vowel marks; conservative collisions share one split. Raw text/model units retain marks. Exact duplicate checks retain marks; no additional fuzzy matching."
    )
    return result


def prepare_data(config, redo_audit=False):
    language = config["language"]
    audit_path = ROOT / "analysis" / f"{language}_audit.jsonl"
    rows = (
        audit(language, config)
        if redo_audit or not audit_path.exists()
        else read_jsonl(audit_path)
    )
    if any(
        row["dataset"] != config["dataset"]["repo"]
        or row["revision"] != config["dataset"]["revision"]
        for row in rows
    ):
        raise ValueError(
            "Cached audit source differs from configured dataset; rebuild with --redo-audit"
        )
    rows = select(rows, config["seed"], **config["selection"])
    write_jsonl(ROOT / "analysis" / f"{language}_selection.jsonl", rows)
    for split in ["train", "val"]:
        write_jsonl(
            ROOT / "data/manifests" / f"{language}_{split}.jsonl",
            [r for r in rows if r["split"] == split],
        )
    summary = summarize(rows, config)
    write_json(ROOT / "analysis" / f"{language}_dataset_summary.json", summary)
    print(json.dumps(summary, indent=2))
