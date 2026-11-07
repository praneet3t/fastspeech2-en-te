"""Summarize actual accepted durations and plot representative held-out alignments."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shared.io import ROOT, read_jsonl, write_json


def audit(language):
    base = ROOT / "data/features" / language
    rows = read_jsonl(base / "train.jsonl") + read_jsonl(base / "val.jsonl")
    durations = []
    silence = 0
    frames = 0
    zero = 0
    units = 0
    for row in rows:
        with np.load(ROOT / row["feature_path"]) as f:
            d = f["duration"]
            mel = f["mel"]
            if len(d) != len(row["units"]) or d.sum() != len(mel) or (d < 0).any():
                raise ValueError(f'Invalid accepted alignment {row["source_id"]}')
            silence += int(sum(n for u, n in zip(row["units"], d) if u == "sil"))
            frames += len(mel)
            zero += int((d == 0).sum())
            units += len(d)
            durations.extend(n for u, n in zip(row["units"], d) if u != "sil")
    report = dict(
        accepted_utterances=len(rows),
        mel_frames=frames,
        units=units,
        zero_duration_units=zero,
        silence_frame_fraction=silence / frames,
        speech_unit_duration_frames_percentiles=np.percentile(
            durations, [0, 25, 50, 75, 95, 99, 100]
        ).tolist(),
        duration_convention="round shared boundaries once, last boundary=floor(samples/256)",
        manual_phone_boundary_accuracy="not established; no hand-labelled alignment benchmark",
    )
    folder = ROOT / "analysis" / language / "alignment_examples"
    folder.mkdir(parents=True, exist_ok=True)
    validation = sorted(
        [r for r in rows if r["split"] == "val"], key=lambda r: r["frames"]
    )
    chosen = [
        validation[len(validation) // 4],
        validation[len(validation) // 2],
        validation[3 * len(validation) // 4],
    ]
    report["examples"] = []
    for row in chosen:
        with np.load(ROOT / row["feature_path"]) as f:
            mel = f["mel"]
            d = f["duration"]
        bounds = np.r_[0, np.cumsum(d)] * 256 / 22050
        fig, ax = plt.subplots(figsize=(max(12, len(d) * 0.17), 4))
        ax.imshow(
            mel.T,
            origin="lower",
            aspect="auto",
            extent=[0, bounds[-1], 0, 80],
            cmap="magma",
        )
        ax.vlines(bounds, 0, 80, color="white", alpha=0.4, linewidth=0.4)
        ax.set_xticks(
            (bounds[:-1] + bounds[1:]) / 2, row["units"], rotation=90, fontsize=6
        )
        ax.secondary_xaxis("top").set_xlabel("Time (s)")
        ax.set(
            xlabel="Aligned linguistic units",
            ylabel="Mel band",
            title=f'{language}: {row["source_id"]}',
        )
        fig.tight_layout()
        path = folder / f'{row["source_id"].replace(":","_")}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        report["examples"].append(
            dict(source_id=row["source_id"], plot=str(path.relative_to(ROOT)))
        )
    write_json(ROOT / "analysis" / language / "alignment_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language", choices=["english", "telugu", "both"], default="both"
    )
    args = parser.parse_args()
    for language in (
        ["english", "telugu"] if args.language == "both" else [args.language]
    ):
        audit(language)
