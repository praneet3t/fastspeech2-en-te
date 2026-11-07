"""Record a final checkpoint separately from the training objective minimum."""

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from shared.io import ROOT, read_jsonl, write_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=["english", "telugu"], required=True)
    parser.add_argument(
        "--criterion", choices=["objective", "validation-probe"], default="objective"
    )
    args = parser.parse_args()
    if not (ROOT / "analysis" / args.language / "training_complete.json").is_file():
        raise ValueError(
            "Wait for training to finish before final checkpoint selection"
        )
    name = args.language.title()
    config_path = ROOT / name / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    folder = ROOT / config["training"]["checkpoint_dir"]
    metrics = [
        r
        for r in read_jsonl(ROOT / "analysis" / args.language / "metrics.jsonl")
        if r["split"] == "val"
    ]
    minimum = min(metrics, key=lambda r: r["total"])
    role = "best"
    probe = None
    if args.criterion == "validation-probe":
        probe = json.loads(
            (ROOT / "analysis" / args.language / "checkpoint_probe.json").read_text()
        )
        role = min(probe["candidates"], key=lambda k: probe["candidates"][k]["asr_wer"])
    source = folder / f"{role}.pt"
    digest = sha256(source)
    if probe and digest != probe["candidates"][role]["checkpoint_sha256"]:
        raise ValueError(
            "Checkpoint changed since the probe; repeat comparison before selection"
        )
    state = torch.load(source, map_location="cpu", weights_only=False)
    selected = next(r for r in metrics if r["step"] == state["step"])
    destination = folder / "selected.pt"
    shutil.copyfile(source, destination)
    criterion = (
        "Lowest teacher-forced validation total"
        if not probe
        else "Lowest Whisper large-v3 WER on a fixed validation-only probe; this is an intelligibility proxy, not naturalness"
    )
    result = dict(
        selected_step=state["step"],
        selected_checkpoint=str(destination.relative_to(ROOT)),
        selected_checkpoint_sha256=digest,
        selected_validation_total=selected["total"],
        minimum_objective_step=minimum["step"],
        minimum_objective_total=minimum["total"],
        criterion_id=args.criterion,
        criterion=criterion,
        supplied_tests_used_for_comparison=False,
    )
    if probe:
        result["validation_probe"] = probe
    write_json(ROOT / "analysis" / args.language / "checkpoint_selection.json", result)
    config["inference_checkpoint"] = str(destination.relative_to(ROOT))
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
