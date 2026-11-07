"""Recompute saved normalization statistics from training measurements only."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from shared.io import ROOT, read_jsonl, write_json

report = {}
for language in ["english", "telugu"]:
    import json

    base = ROOT / "data/features" / language
    stats = json.loads((base / "stats.json").read_text())
    pitch, energy = [], []
    for row in read_jsonl(base / "train.jsonl"):
        path = base / "raw" / f'{row["source_id"].replace(":","_")}.npz'
        with np.load(path) as values:
            pitch.append(values["voiced_frame_f0"])
            energy.append(values["energy"][values["duration"] > 0])
    voiced = np.concatenate(pitch)
    nonempty = np.concatenate(energy)
    expected = [
        float(voiced.mean()),
        float(voiced.std()),
        float(nonempty.mean()),
        float(nonempty.std()),
    ]
    stored = stats["pitch"][2:] + stats["energy"][2:]
    if (voiced <= 0).any() or not np.allclose(stored, expected, rtol=1e-6, atol=1e-6):
        raise ValueError(
            f"{language}: saved normalization differs from train-only measurements"
        )
    report[language] = dict(
        verified=True,
        voiced_train_frames=len(voiced),
        nonempty_train_units=len(nonempty),
        pitch_mean_hz=expected[0],
        pitch_std_hz=expected[1],
        energy_mean=expected[2],
        energy_std=expected[3],
        validation_used=False,
        unvoiced_zero_measurements_used=False,
    )
write_json(ROOT / "analysis/normalization_audit.json", report)
print(report)
