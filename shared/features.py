"""HiFi-GAN-compatible mels, WORLD F0 and aligned phone-level variance targets."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import pyworld
import soundfile as sf
from shared.alignment import read_alignment, resample
from shared.io import ROOT, read_jsonl, write_json, write_jsonl


@lru_cache(maxsize=1)
def mel_filter():
    return torch.from_numpy(
        librosa.filters.mel(
            sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=8000, htk=False, norm="slaney"
        )
    )


def acoustic_features(samples: np.ndarray):
    """Original HiFi-GAN STFT: reflect pad 384, periodic Hann, no centering.

    Returns [frames,80] natural-log mel magnitudes and [frames] L2 energy.
    No peak or loudness normalization is applied.
    """
    if samples.ndim != 1 or len(samples) < 1024 or not np.isfinite(samples).all():
        raise ValueError("Expected finite mono waveform >=1024 samples")
    waveform = torch.from_numpy(samples).float().unsqueeze(0)
    waveform = F.pad(waveform.unsqueeze(1), (384, 384), mode="reflect").squeeze(1)
    spectrum = torch.stft(
        waveform,
        1024,
        256,
        1024,
        window=torch.hann_window(1024),
        center=False,
        return_complex=True,
    )
    magnitude = torch.sqrt(spectrum.real.square() + spectrum.imag.square() + 1e-9)
    mel = torch.log(torch.clamp(mel_filter() @ magnitude, min=1e-5))
    energy = torch.linalg.vector_norm(magnitude, dim=1)
    return mel.squeeze(0).T.numpy(), energy.squeeze(0).numpy()


def phone_average(values, durations):
    boundaries = np.r_[0, np.cumsum(durations)]
    return np.array(
        [
            values[a:b].mean() if b > a else 0.0
            for a, b in zip(boundaries[:-1], boundaries[1:])
        ],
        dtype="float32",
    )


def extract_one(arguments):
    row, config = arguments
    torch.set_num_threads(1)
    language = config["language"]
    try:
        samples, rate = sf.read(
            ROOT / row["audio_path"], dtype="float32", always_2d=True
        )
        samples = resample(samples.mean(1), rate, 22050)
        name = row["source_id"].replace(":", "_")
        grid = (
            ROOT
            / "data/alignment"
            / language
            / f'aligned_{row["split"]}'
            / row["speaker_id"]
            / f"{name}.TextGrid"
        )
        expected = [p for phones in row["pronunciations"] for p in phones]
        units, duration, audit = read_alignment(
            grid, expected, len(samples), 22050, 256
        )
        mel, frame_energy = acoustic_features(samples)
        if len(mel) != duration.sum():
            raise ValueError(
                f"Mel frames {len(mel)} != aligned frames {duration.sum()}"
            )
        f0, times = pyworld.dio(
            samples.astype("float64"),
            22050,
            frame_period=256 / 22050 * 1000,
            f0_floor=50,
            f0_ceil=600,
        )
        f0 = pyworld.stonemask(samples.astype("float64"), f0, times, 22050)
        voiced = f0 > 0
        if voiced.sum() < 2:
            raise ValueError("Fewer than two voiced WORLD F0 frames")
        # Same continuous scalar F0 idea as upstream, NOT a voicing predictor.
        # Interpolation through unvoiced regions is retained and disclosed.
        frame_centers = (np.arange(len(mel)) + 0.5) * 256 / 22050
        continuous = np.interp(frame_centers, times[voiced], f0[voiced])
        voiced_mask = np.interp(frame_centers, times, voiced.astype("float32")) > 0.5
        pitch = phone_average(continuous, duration)
        energy = phone_average(frame_energy, duration)
        if not all(np.isfinite(x).all() for x in (mel, pitch, energy)):
            raise ValueError("Non-finite acoustic features")
        output = ROOT / "data/features" / language / "raw" / f"{name}.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                mel=mel,
                pitch=pitch,
                energy=energy,
                duration=duration,
                units=np.array(units),
                voiced_frame_f0=f0[voiced].astype("float32"),
                voiced_mask=voiced_mask,
            )
        os.replace(temporary, output)
        result = dict(
            row,
            units=units,
            feature_path=str(output.relative_to(ROOT)),
            frames=len(mel),
            alignment_audit=audit,
            voiced_fraction=float(voiced_mask.mean()),
        )
        return result, None
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        return None, dict(
            source_id=row["source_id"],
            split=row["split"],
            reason=str(error),
            duration_seconds=row["duration_seconds"],
        )


def prepare_features(config):
    language = config["language"]
    base = ROOT / "data/features" / language
    selected = sum(
        [
            read_jsonl(ROOT / "data/manifests" / f"{language}_{split}.jsonl")
            for split in ["train", "val"]
        ],
        [],
    )
    accepted, failures = [], []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for index, (result, failure) in enumerate(
            pool.map(extract_one, ((r, config) for r in selected), chunksize=8)
        ):
            if failure:
                failures.append(failure)
            else:
                accepted.append(result)
            if (index + 1) % 200 == 0:
                print(
                    f"{index+1}/{len(selected)} features, {len(failures)} rejected",
                    flush=True,
                )
    write_jsonl(
        ROOT / "analysis" / f"{language}_preprocessing_failures.jsonl", failures
    )
    # Compute all statistics exclusively from training recordings.
    pitches, energies = [], []
    for row in accepted:
        if row["split"] != "train":
            continue
        with np.load(ROOT / row["feature_path"]) as features:
            # Voiced measurements determine F0 normalization; zeros never enter it.
            pitches.append(features["voiced_frame_f0"])
            energies.append(features["energy"][features["duration"] > 0])
    if not pitches:
        raise ValueError(
            "No valid aligned training samples. Inspect preprocessing_failures.jsonl"
        )
    pitch_values, energy_values = np.concatenate(pitches), np.concatenate(energies)
    pitch_mean, pitch_std, energy_mean, energy_std = map(
        float,
        (
            pitch_values.mean(),
            pitch_values.std(),
            energy_values.mean(),
            energy_values.std(),
        ),
    )
    if pitch_std <= 0 or energy_std <= 0:
        raise ValueError("Degenerate normalization statistics")
    pitch_range, energy_range = [], []
    vocab = {"<pad>": 0}
    for unit in sorted({u for r in accepted for u in r["units"]} | {"sil"}):
        vocab[unit] = len(vocab)
    # A validation-only phone would be an untrained input embedding.
    training_units = {u for r in accepted if r["split"] == "train" for u in r["units"]}
    unseen = {
        u for r in accepted if r["split"] == "val" for u in r["units"]
    } - training_units
    if unseen:
        raise ValueError(f"Validation phones missing from training: {unseen}")
    speakers = {
        s: i
        for i, s in enumerate(
            sorted({r["speaker_id"] for r in accepted if r["split"] == "train"})
        )
    }
    for row in accepted:
        with np.load(ROOT / row["feature_path"]) as f:
            values = {k: f[k] for k in ["mel", "pitch", "energy", "duration"]}
        values["pitch"] = (values["pitch"] - pitch_mean) / pitch_std
        values["energy"] = (values["energy"] - energy_mean) / energy_std
        values["tokens"] = np.array([vocab[u] for u in row["units"]], dtype="int64")
        out = base / f'{row["source_id"].replace(":","_")}.npz'
        temporary = out.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, **values)
        os.replace(temporary, out)
        row["feature_path"] = str(out.relative_to(ROOT))
        if row["split"] == "train":
            pitch_range.extend(
                [float(values["pitch"].min()), float(values["pitch"].max())]
            )
            energy_range.extend(
                [float(values["energy"].min()), float(values["energy"].max())]
            )
    write_json(
        base / "stats.json",
        dict(
            pitch=[min(pitch_range), max(pitch_range), pitch_mean, pitch_std],
            energy=[min(energy_range), max(energy_range), energy_mean, energy_std],
            pitch_representation="WORLD Hz, linear interpolation across unvoiced frames, averaged per aligned unit; voiced train frames for normalization",
            energy_representation="L2 STFT magnitude norm averaged per aligned unit, standardized on train only",
        ),
    )
    write_json(base / "vocab.json", vocab)
    write_json(base / "speakers.json", speakers)
    report = dict(
        selected=len(selected),
        accepted=len(accepted),
        rejected=len(failures),
        failures=dict(Counter(f["reason"].split(":")[0] for f in failures)),
        splits={},
    )
    for split in ["train", "val"]:
        rows = [r for r in accepted if r["split"] == split]
        write_jsonl(base / f"{split}.jsonl", rows)
        report["splits"][split] = dict(
            utterances=len(rows),
            hours=sum(r["duration_seconds"] for r in rows) / 3600,
            mel_frames=sum(r["frames"] for r in rows),
            zero_duration_units=sum(
                r["alignment_audit"]["zero_duration_units"] for r in rows
            ),
        )
    write_json(ROOT / "analysis" / f"{language}_preprocessing.json", report)
    print(json.dumps(report, indent=2))
    if (
        report["splits"]["train"]["hours"] < 4.75
        or report["splits"]["val"]["hours"] < 0.95
    ):
        raise ValueError(
            "Post-alignment duration below assignment tolerance. Repair failed alignments before training."
        )
