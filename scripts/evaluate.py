"""Held-out objective/variance errors and free-running length sanity checks."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from shared.io import ROOT, load_config, write_json, sha256, inference_checkpoint
from shared.model import build_model
from shared.batches import Features, collate, to_device
from shared.train import validate
from shared.features import phone_average
from shared.text import frontend
from shared.inference import inference_units


@torch.inference_mode()
def evaluate(language):
    config = load_config(ROOT / language.title() / "config.yaml")
    checkpoint = inference_checkpoint(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = state["metadata"]
    model, criterion = build_model(config, metadata)
    model.load_state_dict(state["model"])
    model.cuda().eval()
    criterion.cuda()
    data = Features(language, "val")
    loader = DataLoader(
        data, batch_size=8, collate_fn=collate, shuffle=False, num_workers=2
    )
    results = validate(model, criterion, loader, "cuda")
    model.eval()
    pitch_error = 0.0
    pitch_count = 0
    energy_error = 0.0
    energy_count = 0
    stats = metadata["stats"]
    rows = {r["source_id"]: r for r in data.rows}
    for cpu_batch in loader:
        batch = to_device(cpu_batch, "cuda")
        output = model(*batch[2:])
        p = output[2].cpu().numpy()
        e = output[3].cpu().numpy()
        for index, source_id in enumerate(batch[0]):
            n = int(batch[4][index])
            row = rows[source_id]
            raw = (
                ROOT
                / "data/features"
                / language
                / "raw"
                / f'{source_id.replace(":","_")}.npz'
            )
            with np.load(raw) as values:
                voiced_fraction = phone_average(
                    values["voiced_mask"].astype("float32"), values["duration"]
                )
                duration = values["duration"]
            voiced = (voiced_fraction >= 0.8) & (duration > 0)
            nonempty = duration > 0
            pitch_difference = (p[index, :n] - cpu_batch[9][index, :n].numpy()) * stats[
                "pitch"
            ][3]
            energy_difference = (
                e[index, :n] - cpu_batch[10][index, :n].numpy()
            ) * stats["energy"][3]
            pitch_error += float((pitch_difference[voiced] ** 2).sum())
            pitch_count += int(voiced.sum())
            energy_error += float((energy_difference[nonempty] ** 2).sum())
            energy_count += int(nonempty.sum())
    results.update(
        total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        pitch_rmse_hz_mostly_voiced_units=(
            (pitch_error / pitch_count) ** 0.5 if pitch_count else None
        ),
        pitch_evaluated_units=pitch_count,
        energy_rmse_stft_l2=(energy_error / energy_count) ** 0.5,
        pitch_evaluation="continuous target, units with >=80% voiced frames and nonzero duration; teacher-forced path",
        checkpoint_step=state["step"],
        checkpoint_sha256=sha256(checkpoint),
        language=language,
    )
    free = []
    for row in data.rows[:12]:
        text = frontend(row["raw_text"], language)
        units = inference_units(text)
        missing = sorted(set(units) - metadata["vocab"].keys())
        if missing:
            free.append(
                dict(source_id=row["source_id"], error=f"Unseen phones: {missing}")
            )
            continue
        tokens = torch.tensor([[metadata["vocab"][u] for u in units]], device="cuda")
        try:
            output = model(
                torch.tensor([metadata["speakers"][row["speaker_id"]]], device="cuda"),
                tokens,
                torch.tensor([len(units)], device="cuda"),
                len(units),
            )
            frames = int(output[9][0])
            reference = row["frames"]
            free.append(
                dict(
                    source_id=row["source_id"],
                    predicted_frames=frames,
                    reference_frames=reference,
                    duration_ratio=frames / reference,
                )
            )
        except ValueError as error:
            free.append(dict(source_id=row["source_id"], error=str(error)))
    results["free_running_validation_lengths"] = free
    write_json(ROOT / "analysis" / language / "evaluation.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language", choices=["english", "telugu", "both"], default="both"
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    for language in (
        ["english", "telugu"] if args.language == "both" else [args.language]
    ):
        evaluate(language)
