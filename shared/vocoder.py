"""Pinned upstream universal HiFi-GAN, with explicit conditioning checks."""

import json
import sys
import torch
from shared.io import ROOT

vendor = str(ROOT / "vendor/fastspeech2")
if vendor not in sys.path:
    sys.path.insert(0, vendor)
import hifigan  # noqa: E402


def load_vocoder(config, device):
    path = ROOT / config["vocoder"]["checkpoint"]
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing HiFi-GAN checkpoint {path}. Run python scripts/download_vocoder.py"
        )
    h = json.loads((ROOT / config["vocoder"]["config"]).read_text())
    expected = {
        "sampling_rate": 22050,
        "n_fft": 1024,
        "hop_size": 256,
        "win_size": 1024,
        "num_mels": 80,
        "fmin": 0,
        "fmax": 8000,
    }
    if any(h.get(k) != v for k, v in expected.items()):
        raise ValueError(
            "Vocoder acoustic configuration is incompatible with extracted features"
        )
    if (
        config["audio"]["sample_rate"] != h["sampling_rate"]
        or config["audio"]["hop_length"] != h["hop_size"]
    ):
        raise ValueError("Acoustic model and vocoder rates/hops differ")
    vocoder = hifigan.Generator(hifigan.AttrDict(h))
    vocoder.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=False)["generator"]
    )
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)
    return vocoder


@torch.inference_mode()
def vocode(vocoder, mel):
    if (
        mel.ndim != 3
        or mel.shape[1] != 80
        or mel.shape[2] < 1
        or not torch.isfinite(mel).all()
    ):
        raise ValueError(
            "Vocoder input must be finite [batch,80,frames] natural-log magnitude mels"
        )
    wave = vocoder(mel).squeeze(1)
    if wave.shape[1] != mel.shape[2] * 256 or not torch.isfinite(wave).all():
        raise ValueError("Invalid HiFi-GAN waveform shape or values")
    if wave.abs().max() < 1e-5:
        raise ValueError("HiFi-GAN generated silence")
    return wave
