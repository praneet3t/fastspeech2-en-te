"""Load the retained upstream model and its small, serializable input metadata."""

import json
import sys
import tempfile
from pathlib import Path
from shared.io import ROOT, read_jsonl, sha256

VENDOR = ROOT / "vendor/fastspeech2"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
from model import FastSpeech2, FastSpeech2Loss  # noqa: E402


def load_metadata(config):
    base = ROOT / "data/features" / config["language"]
    metadata = {}
    for name in ["stats", "vocab", "speakers"]:
        path = base / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}; complete real alignment and preprocessing first"
            )
        metadata[name] = json.loads(path.read_text())
    hours = {speaker: 0.0 for speaker in metadata["speakers"]}
    for row in read_jsonl(base / "train.jsonl"):
        hours[row["speaker_id"]] += row["duration_seconds"] / 3600
    metadata["speaker_training_hours"] = hours
    metadata["feature_hashes"] = {
        name: sha256(base / name)
        for name in [
            "stats.json",
            "vocab.json",
            "speakers.json",
            "train.jsonl",
            "val.jsonl",
        ]
    }
    return metadata


def build_model(config, metadata=None):
    metadata = load_metadata(config) if metadata is None else metadata
    # Upstream constructors read two small JSON files. A temporary directory
    # retains that interface while making saved checkpoints corpus-independent.
    with tempfile.TemporaryDirectory() as directory:
        for name in ["stats", "speakers"]:
            (Path(directory) / f"{name}.json").write_text(json.dumps(metadata[name]))
        preprocess = dict(
            path=dict(preprocessed_path=directory),
            preprocessing=dict(
                mel=dict(n_mel_channels=80),
                pitch=dict(feature="phoneme_level"),
                energy=dict(feature="phoneme_level"),
            ),
        )
        model_config = dict(config["model"])
        model_config["vocab_size"] = len(metadata["vocab"])
        model = FastSpeech2(preprocess, model_config)
        criterion = FastSpeech2Loss(preprocess, model_config)
    return model, criterion
