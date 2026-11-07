"""Rebuild all figures from persisted real metrics and model prediction JSON."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shared.io import ROOT, read_jsonl


def plots(language):
    name = language.title()
    result = ROOT / "Results" / name
    result.mkdir(parents=True, exist_ok=True)
    metrics = read_jsonl(ROOT / "analysis" / language / "metrics.jsonl")
    for split, filename in [
        ("train", "training_loss.png"),
        ("val", "validation_loss.png"),
    ]:
        rows = [r for r in metrics if r["split"] == split]
        if not rows:
            raise ValueError(f"No actual {language} {split} losses")
        fig, ax = plt.subplots(figsize=(9, 5))
        for key in ["total", "mel", "postnet_mel", "duration", "pitch", "energy"]:
            ax.plot(
                [r["step"] for r in rows],
                [r[key] for r in rows],
                label=key,
                linewidth=1,
            )
        ax.set(
            xlabel="Optimizer step",
            ylabel="Loss (component units differ)",
            title=f"{name}: {split}" + (" (teacher-forced)" if split == "val" else ""),
        )
        ax.legend(ncol=3)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(result / filename, dpi=150)
        plt.close(fig)
    prediction_files = sorted(
        (ROOT / "analysis" / language / "predictions").glob("*.json")
    )
    if len(prediction_files) != 5:
        raise ValueError(
            f"Expected five {language} actual test predictions, found {len(prediction_files)}"
        )
    for path in prediction_files:
        prediction = json.loads(path.read_text())
        units = prediction["units"]
        x = np.arange(len(units))
        for kind, key, ylabel in [
            ("duration", "duration_ms", "Predicted duration (ms)"),
            (
                "pitch",
                "pitch_continuous_hz",
                "Predicted continuous F0 (Hz; unit-level)",
            ),
            (
                "energy",
                "energy_stft_l2",
                "Predicted energy (STFT magnitude L2; unit-level)",
            ),
        ]:
            folder = result / "variance_adaptor_plots" / kind
            folder.mkdir(parents=True, exist_ok=True)
            # Consecutive panels keep every unit readable at report page width.
            per_panel = 36
            panels = (len(units) + per_panel - 1) // per_panel
            fig, axes = plt.subplots(
                panels,
                1,
                figsize=(10, 2.5 * panels + 0.5),
                squeeze=False,
                sharey=True,
            )
            values = np.asarray(prediction[key])
            for panel, ax in enumerate(axes[:, 0]):
                start = panel * per_panel
                end = min(start + per_panel, len(units))
                indices = x[start:end]
                if kind == "duration":
                    ax.bar(indices, values[start:end], width=0.8)
                else:
                    ax.plot(indices, values[start:end], marker=".", linewidth=1)
                ax.set_xticks(
                    indices,
                    [f"{i+1}:{units[i]}" for i in indices],
                    rotation=90,
                    fontsize=9,
                )
                ax.set_xlim(start - 0.7, end - 0.3)
                ax.set_title(f"Units {start+1}–{end}", fontsize=9, loc="left")
                ax.grid(axis="y", alpha=0.2)
            fig.suptitle(
                f'{prediction["test_id"]} — {name}; checkpoint step {prediction["checkpoint_step"]}',
                fontsize=12,
            )
            fig.supxlabel(
                "Consecutive model units (1-based index; sil = silence)", fontsize=10
            )
            fig.supylabel(ylabel, fontsize=10)
            fig.tight_layout(rect=(0.025, 0.015, 1, 0.97))
            fig.savefig(folder / f'{prediction["test_id"]}.png', dpi=150)
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language", choices=["english", "telugu", "both"], default="both"
    )
    args = parser.parse_args()
    for language in (
        ["english", "telugu"] if args.language == "both" else [args.language]
    ):
        plots(language)
