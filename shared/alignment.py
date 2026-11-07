"""MFA corpus preparation and auditable interval-to-frame conversion."""

import hashlib
import os
from pathlib import Path
import re
import subprocess
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
from praatio import textgrid
from shared.io import ROOT, read_jsonl, write_json


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples.astype("float32")
    divisor = gcd(source_rate, target_rate)
    return resample_poly(
        samples, target_rate // divisor, source_rate // divisor
    ).astype("float32")


def lexical_id(word):
    # Opaque lowercase words prevent MFA's punctuation/text cleaning from changing
    # the exact word-to-phone inventory. words.json retains the reversible mapping.
    return "w" + hashlib.sha256(word.encode()).hexdigest()[:24]


def prepare_corpus(config):
    language = config["language"]
    base = ROOT / "data/alignment" / language
    dictionary, word_map = {}, {}
    for split in ["train", "val"]:
        for row in read_jsonl(ROOT / "data/manifests" / f"{language}_{split}.jsonl"):
            name = row["source_id"].replace(":", "_")
            folder = base / split / row["speaker_id"]
            folder.mkdir(parents=True, exist_ok=True)
            words = []
            for word, phones in zip(row["words"], row["pronunciations"]):
                if phones in (["sp"], ["sil"]):
                    continue  # MFA's optional silence models punctuation pauses acoustically.
                key = lexical_id(word)
                pronunciation = tuple("sil" if p == "sp" else p for p in phones)
                if key in dictionary and dictionary[key] != pronunciation:
                    raise ValueError(f"Inconsistent dictionary entry: {word}")
                dictionary[key] = pronunciation
                word_map[key] = word
                words.append(key)
            wav = folder / f"{name}.wav"
            if not wav.exists():
                samples, rate = sf.read(
                    ROOT / row["audio_path"], dtype="float32", always_2d=True
                )
                temporary = wav.with_suffix(".tmp")
                sf.write(
                    temporary,
                    resample(samples.mean(1), rate, 16000),
                    16000,
                    subtype="PCM_16",
                    format="WAV",
                )
                os.replace(temporary, wav)
            (folder / f"{name}.lab").write_text(" ".join(words), encoding="utf-8")
    (base / "dictionary.txt").write_text(
        "".join(f'{w}\t{" ".join(p)}\n' for w, p in sorted(dictionary.items())),
        encoding="utf-8",
    )
    write_json(base / "words.json", word_map)
    return base


def run_mfa(config, train_acoustic=False, clean=False):
    base = prepare_corpus(config)
    exe = config["alignment"]["mfa_executable"]
    jobs = str(config["alignment"]["jobs"])
    model = config["alignment"]["acoustic_model"]
    if config["language"] == "telugu":
        model = str(ROOT / model)
        if train_acoustic:
            Path(model).parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    exe,
                    "train",
                    str(base / "train"),
                    str(base / "dictionary.txt"),
                    model,
                    "--output_directory",
                    str(base / "aligned_train"),
                    "--config_path",
                    str(ROOT / "references/mfa_telugu.yaml"),
                    "--num_jobs",
                    jobs,
                    "--temporary_directory",
                    str(base / "work_train"),
                    "--clean" if clean else "--no_clean",
                    "--overwrite",
                ],
                check=True,
            )
        if not Path(model).exists():
            raise FileNotFoundError(
                "No Telugu acoustic aligner. Run preprocessing with --stage align --train-aligner."
            )
    for split in ["train", "val"]:
        if train_acoustic and split == "train" and config["language"] == "telugu":
            continue
        subprocess.run(
            [
                exe,
                "align",
                str(base / split),
                str(base / "dictionary.txt"),
                model,
                str(base / f"aligned_{split}"),
                "--num_jobs",
                jobs,
                "--clean" if clean else "--no_clean",
                "--overwrite",
                "--beam",
                "100",
                "--retry_beam",
                "400",
                "--temporary_directory",
                str(base / f"work_{split}"),
            ]
            + (["--single_speaker"] if config["language"] == "english" else []),
            check=True,
        )


def read_alignment(
    path: Path,
    expected_phones: list[str],
    sample_count: int,
    sample_rate: int,
    hop: int,
):
    """Round shared boundaries once; retain leading/trailing silence.

    HiFi-GAN's center=False, reflect-padded STFT yields floor(samples / hop)
    frames. The last boundary is clamped to exactly that number. Empty gaps in
    MFA's phone tier become silence. Non-silence labels must match the dictionary.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing forced alignment: {path}")
    grid = textgrid.openTextgrid(str(path), includeEmptyIntervals=True)
    tiers = [name for name in grid.tierNames if name.endswith("phones")]
    if len(tiers) != 1:
        raise ValueError(f"Expected one phone tier, got {tiers}")
    entries = grid.getTier(tiers[0]).entries
    if not entries:
        raise ValueError("Empty phone alignment")
    total_frames = sample_count // hop
    units, starts, ends = [], [], []
    previous = 0.0
    total_seconds = sample_count / sample_rate
    for entry in entries:
        start, end, label = entry
        if start < previous - 0.001 or end < start or end > total_seconds + 0.05:
            raise ValueError("Non-monotonic/out-of-bounds phone interval")
        if start > previous + 0.001:
            units.append("sil")
            starts.append(previous)
            ends.append(start)
        label = re.sub(r"_[BIES]$", "", label.strip())
        label = "sil" if label in ("", "sil", "sp") else label
        if label in ("spn", "<unk>"):
            raise ValueError("Unrecognized speech interval in forced alignment")
        units.append(label)
        starts.append(start)
        ends.append(end)
        previous = end
    if previous < total_seconds - 0.001:
        units.append("sil")
        starts.append(previous)
        ends.append(total_seconds)
    # Silence placement is acoustic, so compare only speech phone sequence.
    expected = [p for p in expected_phones if p not in ("sil", "sp")]
    actual = [p for p in units if p != "sil"]
    if actual != expected:
        raise ValueError(
            f"Alignment phone sequence mismatch: expected {expected[:20]}, got {actual[:20]}"
        )
    boundaries = np.rint(np.array([0.0] + ends) * sample_rate / hop).astype("int64")
    boundaries = np.clip(boundaries, 0, total_frames)
    boundaries[-1] = total_frames
    durations = np.diff(boundaries)
    if (
        (durations < 0).any()
        or durations.sum() != total_frames
        or len(durations) != len(units)
    ):
        raise ValueError("Invalid frame allocation")
    speech_seconds = [e - s for u, s, e in zip(units, starts, ends) if u != "sil"]
    if not speech_seconds or max(speech_seconds) > 2.0:
        raise ValueError("Implausible speech phone >2 seconds or no speech")
    return (
        units,
        durations,
        dict(
            frames=total_frames,
            zero_duration_units=int((durations == 0).sum()),
            silence_frames=int(sum(d for u, d in zip(units, durations) if u == "sil")),
            longest_speech_phone_seconds=max(speech_seconds),
        ),
    )
