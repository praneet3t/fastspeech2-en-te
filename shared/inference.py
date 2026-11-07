"""Synthesize supplied tests; export actual predicted variances and WAV checks."""

import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from shared.io import ROOT, load_config, write_json, sha256, inference_checkpoint
from shared.text import frontend
from shared.model import build_model, load_metadata
from shared.vocoder import load_vocoder, vocode


def inference_units(text):
    units = ["sil"]
    for phone in text.units:
        phone = "sil" if phone == "sp" else phone
        if phone != "sil" or units[-1] != "sil":
            units.append(phone)
    if units[-1] != "sil":
        units.append("sil")
    return units


def check_audio(path, expected_rate=22050):
    samples, rate = sf.read(path, dtype="float32")
    if (
        rate != expected_rate
        or samples.ndim != 1
        or len(samples) == 0
        or not np.isfinite(samples).all()
    ):
        raise ValueError(f"Invalid generated WAV: {path}")
    rms = float(np.sqrt(np.mean(samples.astype("float64") ** 2)))
    clipped = float(np.mean(np.abs(samples) >= 0.999))
    if rms < 1e-5:
        raise ValueError(f"Completely/near silent generated WAV: {path}")
    if clipped > 0.01:
        raise ValueError(f"Severe clipping in generated WAV: {path}")
    return dict(
        sample_rate=rate,
        samples=len(samples),
        duration_seconds=len(samples) / rate,
        rms=rms,
        peak=float(np.max(np.abs(samples))),
        clipped_fraction=clipped,
    )


def word_durations(text, units, durations):
    """Trace word boundaries through inference units; durations are mel frames."""
    word_predictions = []
    cursor = 1
    for word, phones in zip(text.words, text.pronunciations):
        phones = ["sil" if p == "sp" else p for p in phones]
        if phones == ["sil"]:
            if cursor < len(units) and units[cursor] == "sil":
                cursor += 1
            continue
        if units[cursor : cursor + len(phones)] != phones:
            raise ValueError("Word/unit tracing mismatch")
        end = cursor + len(phones)
        word_predictions.append(
            dict(
                word=word,
                unit_start=cursor,
                unit_end=end,
                duration_ms=float(durations[cursor:end].sum() * 256 / 22050 * 1000),
            )
        )
        cursor = end
    return word_predictions


@torch.inference_mode()
def synthesize(config, manifest_path, checkpoint_path=None, output_root=None):
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    language = config["language"]
    name = language.title()
    checkpoint = (
        Path(checkpoint_path) if checkpoint_path else inference_checkpoint(config)
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing trained {language} checkpoint: {checkpoint}. Train first; random weights will not be synthesized."
        )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_config = state["config"]
    for key in ["language", "audio", "model"]:
        if saved_config[key] != config[key]:
            raise ValueError(
                f"Checkpoint {key} differs from the supplied configuration; use its recorded configuration"
            )
    metadata = state.get("metadata") or load_metadata(config)
    model, _ = build_model(config, metadata)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    vocoder = load_vocoder(config, device)
    vocab = metadata["vocab"]
    stats = metadata["stats"]
    speakers = metadata["speakers"]
    # Deterministically use the labelled speaker with most accepted train audio.
    hours = metadata["speaker_training_hours"]
    default_speaker = sorted(hours, key=lambda s: (-hours[s], s))[0]
    records = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    output_base = ROOT / output_root if output_root else None
    analysis = (
        output_base / language / "predictions"
        if output_base
        else ROOT / "analysis" / language / "predictions"
    )
    analysis.mkdir(parents=True, exist_ok=True)
    wav_dir = (
        output_base / name if output_base else ROOT / "Results" / name
    ) / "test_generated_audios"
    wav_dir.mkdir(parents=True, exist_ok=True)
    audit = []
    for record in records:
        if record["language"] != language:
            continue
        speaker = record.get("speaker_id") or default_speaker
        if speaker not in speakers:
            raise ValueError(
                f"Unknown speaker {speaker!r}; available: {sorted(speakers)}"
            )
        text = frontend(record["original_text"], language)
        units = inference_units(text)
        missing = sorted(set(units) - vocab.keys())
        if missing:
            raise ValueError(
                f'{record["test_id"]}: inference phones absent from training vocabulary: {missing}'
            )
        ids = torch.tensor([[vocab[u] for u in units]], device=device)
        lengths = torch.tensor([len(units)], device=device)
        output = model(
            torch.tensor([speakers[speaker]], device=device), ids, lengths, len(units)
        )
        mel = output[1].transpose(1, 2)  # [B,T,80] -> vocoder [B,80,T]
        wave = vocode(vocoder, mel)[0].cpu().numpy()
        if float(np.mean(np.abs(wave) >= 1)) > 0.01:
            raise ValueError(
                "Vocoder output would be severely clipped on PCM conversion"
            )
        path = wav_dir / f'{record["test_id"]}.wav'
        sf.write(path, np.clip(wave, -1, 1), 22050, subtype="PCM_16")
        quality = check_audio(path)
        durations = output[5][0, : len(units)].cpu().numpy().astype("int64")
        pitch = output[2][0, : len(units)].float().cpu().numpy()
        energy = output[3][0, : len(units)].float().cpu().numpy()
        prediction = dict(
            test_id=record["test_id"],
            units=units,
            duration_frames=durations.tolist(),
            duration_ms=(durations * 256 / 22050 * 1000).tolist(),
            log_duration=output[4][0, : len(units)].float().cpu().tolist(),
            pitch_normalized=pitch.tolist(),
            pitch_continuous_hz=(
                pitch * stats["pitch"][3] + stats["pitch"][2]
            ).tolist(),
            energy_normalized=energy.tolist(),
            energy_stft_l2=(energy * stats["energy"][3] + stats["energy"][2]).tolist(),
            feature_level="aligned linguistic unit; scalar continuous F0 interpolated through unvoiced regions during target extraction",
            sample_rate=22050,
            hop_length=256,
            speaker_id=speaker,
            speaker_training_hours=hours[speaker],
            speaker_choice=(
                "Most represented accepted training speaker; English is unconditioned mixture and has no selectable identity"
                if language == "english"
                else (
                    "Requested manifest speaker"
                    if record.get("speaker_id")
                    else "Most represented accepted training speaker"
                )
            ),
            checkpoint_step=state.get("step"),
            checkpoint_sha256=sha256(checkpoint),
            vocoder_sha256=sha256(ROOT / config["vocoder"]["checkpoint"]),
            audio=quality,
            original_text=record["original_text"],
            normalization_log=text.transformations,
        )
        prediction["word_predictions"] = word_durations(text, units, durations)
        prediction["zero_duration_units"] = [
            dict(index=i, unit=u)
            for i, (u, d) in enumerate(zip(units, durations))
            if d == 0
        ]
        write_json(analysis / f'{record["test_id"]}.json', prediction)
        record.update(
            internal_normalized_text=text.normalized,
            model_input_sequence=units,
            normalization_log=text.transformations,
            speaker_id=speaker,
            model_checkpoint=(
                str(checkpoint.relative_to(ROOT))
                if checkpoint.is_relative_to(ROOT)
                else str(checkpoint)
            ),
            model_checkpoint_step=state.get("step"),
            vocoder_checkpoint=config["vocoder"]["checkpoint"],
            output_audio_filename=(
                str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
            ),
        )
        audit.append(dict(test_id=record["test_id"], **quality))
        print(record["test_id"], quality, flush=True)
    write_json(manifest_path, records)
    audit_path = (
        output_base / language / "audio_audit.json"
        if output_base
        else ROOT / "analysis" / language / "audio_audit.json"
    )
    write_json(audit_path, audit)


def main(default_language):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=f"{default_language}/config.yaml")
    parser.add_argument("--test-manifest", default="data/test_sentences.json")
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--output-root", help="Keep interim audio/analysis separate from final Results"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if config["language"] != default_language.lower():
        raise ValueError("Entry point/config mismatch")
    synthesize(config, args.test_manifest, args.checkpoint, args.output_root)
