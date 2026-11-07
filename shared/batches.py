"""Feature validation and padded batches in the exact upstream loss interface."""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from shared.io import ROOT, read_jsonl


class Features(Dataset):
    def __init__(self, language, split):
        base = ROOT / "data/features" / language
        self.rows = read_jsonl(base / f"{split}.jsonl")
        self.speakers = json.loads((base / "speakers.json").read_text())
        if not self.rows:
            raise ValueError("Empty feature split")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with np.load(ROOT / row["feature_path"]) as f:
            item = {k: f[k] for k in ["tokens", "mel", "pitch", "energy", "duration"]}
        n = len(item["tokens"])
        if not (
            n == len(item["duration"]) == len(item["pitch"]) == len(item["energy"])
        ):
            raise ValueError(f'Unit/variance length mismatch: {row["source_id"]}')
        if item["duration"].sum() != len(item["mel"]) or (item["duration"] < 0).any():
            raise ValueError(f'Invalid duration targets: {row["source_id"]}')
        if not all(np.isfinite(v).all() for v in item.values()):
            raise ValueError(f'Non-finite features: {row["source_id"]}')
        item.update(id=row["source_id"], speaker=self.speakers[row["speaker_id"]])
        return item


def collate(items):
    """[B,N] tokens/variance; [B,T,80] mel; padding token=0."""
    lengths = torch.tensor([len(x["tokens"]) for x in items], dtype=torch.long)
    mel_lengths = torch.tensor([len(x["mel"]) for x in items], dtype=torch.long)
    max_tokens, max_frames = int(lengths.max()), int(mel_lengths.max())

    def padded(key, max_length, dtype):
        values = [torch.as_tensor(x[key], dtype=dtype) for x in items]
        shape = (len(items), max_length) + values[0].shape[1:]
        result = torch.zeros(shape, dtype=dtype)
        for i, value in enumerate(values):
            result[i, : len(value)] = value
        return result

    return (
        [x["id"] for x in items],
        None,
        torch.tensor([x["speaker"] for x in items], dtype=torch.long),
        padded("tokens", max_tokens, torch.long),
        lengths,
        max_tokens,
        padded("mel", max_frames, torch.float32),
        mel_lengths,
        max_frames,
        padded("pitch", max_tokens, torch.float32),
        padded("energy", max_tokens, torch.float32),
        padded("duration", max_tokens, torch.long),
    )


def to_device(batch, device):
    return tuple(
        x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x
        for x in batch
    )
