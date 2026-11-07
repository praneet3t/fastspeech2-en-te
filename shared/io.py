"""Small file/config helpers. All configured paths are relative to the repository."""

import hashlib
import json
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def inference_checkpoint(config):
    """Final selection may differ from the minimum teacher-forced objective."""
    return ROOT / config.get(
        "inference_checkpoint", config["training"]["checkpoint_dir"] + "/best.pt"
    )


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if config["audio"] != dict(
        sample_rate=22050,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        fmin=0,
        fmax=8000,
    ):
        raise ValueError(
            "Acoustic configuration differs from the pinned HiFi-GAN recipe."
        )
    return config


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
