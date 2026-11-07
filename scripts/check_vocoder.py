"""Reconstruct real held-out mels before trusting a pretrained vocoder.

Numerical reconstruction checks do not establish naturalness. Derived reference
WAVs are local-only dataset material and are never committed.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
import soundfile as sf
import pyworld
from shared.io import ROOT, load_config, read_jsonl, write_json
from shared.features import acoustic_features
from shared.alignment import resample
from shared.vocoder import load_vocoder, vocode
from shared.inference import check_audio

torch.set_num_threads(4)
config = load_config(ROOT / "Telugu/config.yaml")
device = torch.device("cuda")
vocoder = load_vocoder(config, device)
folder = ROOT / "artifacts/vocoder_validation"
folder.mkdir(parents=True, exist_ok=True)
results = []
for language in ["english", "telugu"]:
    rows = read_jsonl(ROOT / "data/manifests" / f"{language}_val.jsonl")
    seen = set()
    for row in rows:
        key = (row["speaker_id"], row["sample_rate"])
        if key in seen or not 3 < row["duration_seconds"] < 8:
            continue
        seen.add(key)
        y, rate = sf.read(ROOT / row["audio_path"], dtype="float32", always_2d=True)
        y = resample(y.mean(1), rate, 22050)
        mel, _ = acoustic_features(y)
        wave = (
            vocode(vocoder, torch.from_numpy(mel.T).unsqueeze(0).to(device))[0]
            .cpu()
            .numpy()
        )
        if np.mean(np.abs(wave) >= 1) > 0.01:
            raise ValueError("Reference mel reconstruction severely clipped")
        path = folder / f'{language}_{row["speaker_id"]}_{rate}.wav'
        sf.write(path, wave, 22050)
        reconstructed, _ = acoustic_features(wave)
        length = min(len(mel), len(reconstructed))
        f, ts = pyworld.dio(y.astype("float64"), 22050, f0_floor=50, f0_ceil=600)
        f = pyworld.stonemask(y.astype("float64"), f, ts, 22050)
        g, ts = pyworld.dio(wave.astype("float64"), 22050, f0_floor=50, f0_ceil=600)
        g = pyworld.stonemask(wave.astype("float64"), g, ts, 22050)
        common_length = min(len(f), len(g))
        voiced_in_both = (f[:common_length] > 0) & (g[:common_length] > 0)
        common_error = np.abs(f[:common_length] - g[:common_length])[voiced_in_both]
        entry = dict(
            language=language,
            source_id=row["source_id"],
            speaker_id=row["speaker_id"],
            source_rate=rate,
            audio=check_audio(path),
            mel_reanalysis_l1=float(
                np.mean(np.abs(mel[:length] - reconstructed[:length]))
            ),
            reference_voiced_f0_median_hz=(
                float(np.median(f[f > 0])) if (f > 0).any() else None
            ),
            reconstructed_voiced_f0_median_hz=(
                float(np.median(g[g > 0])) if (g > 0).any() else None
            ),
            common_voiced_f0_mae_hz=(
                float(common_error.mean()) if len(common_error) else None
            ),
            common_voiced_f0_median_absolute_error_hz=(
                float(np.median(common_error)) if len(common_error) else None
            ),
            common_voiced_frames=int(voiced_in_both.sum()),
            note="Numerical analysis only; no qualified listening assessment",
        )
        results.append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
write_json(ROOT / "analysis/vocoder_validation.json", results)
