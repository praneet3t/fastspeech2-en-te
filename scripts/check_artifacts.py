"""Validate contents, not just names. Returns nonzero if any required item is missing."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
from PIL import Image
from shared.io import ROOT, write_json, load_config, sha256, inference_checkpoint
from shared.inference import check_audio


def audit():
    torch.set_num_threads(4)
    checks = []

    def check(label, function):
        try:
            detail = function()
            checks.append(dict(item=label, status="verified", detail=detail))
        except Exception as error:
            checks.append(
                dict(item=label, status="missing_or_invalid", detail=str(error))
            )

    def assignment_check():
        source = json.loads(
            (ROOT / "analysis/assignment_source.json").read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (ROOT / "data/test_sentences.json").read_text(encoding="utf-8")
        )
        if {r["test_id"] for r in manifest} != set(source["sentences"]):
            raise ValueError("Missing or unexpected assignment sentence")
        for row in manifest:
            if (
                row["original_text"] != source["sentences"][row["test_id"]]
                or not row["pdf_visually_verified"]
            ):
                raise ValueError("Test input differs from verified PDF transcription")
            prediction = json.loads(
                (
                    ROOT
                    / "analysis"
                    / row["language"]
                    / "predictions"
                    / f"{row['test_id']}.json"
                ).read_text(encoding="utf-8")
            )
            if prediction["original_text"] != row["original_text"]:
                raise ValueError("Prediction belongs to an older input")
        return dict(sentences=len(manifest), source_pdf_sha256=source["source_sha256"])

    check("Assignment source and current predictions", assignment_check)

    for language, prefix in [("English", "E"), ("Telugu", "T")]:
        ids = [prefix + "1", prefix + "2a", prefix + "2b", prefix + "3", prefix + "4"]
        for test_id in ids:

            def waveform_check(name=language, test=test_id):
                path = ROOT / "Results" / name / "test_generated_audios" / f"{test}.wav"
                result = check_audio(path)
                prediction = json.loads(
                    (
                        ROOT
                        / "analysis"
                        / name.lower()
                        / "predictions"
                        / f"{test}.json"
                    ).read_text()
                )
                units = prediction["units"]
                for key in ["duration_frames", "pitch_continuous_hz", "energy_stft_l2"]:
                    values = np.asarray(prediction[key])
                    if len(values) != len(units) or not np.isfinite(values).all():
                        raise ValueError(f"Invalid exported {key}")
                durations = np.asarray(prediction["duration_frames"])
                if (durations < 0).any() or not np.equal(
                    durations, durations.astype(int)
                ).all():
                    raise ValueError("Non-integer/negative generated durations")
                if int(durations.sum()) * prediction["hop_length"] != result["samples"]:
                    raise ValueError("Predicted duration sum disagrees with WAV length")
                return result

            check(
                f"{test_id} WAV",
                waveform_check,
            )
            for kind in ["duration", "pitch", "energy"]:
                path = (
                    ROOT
                    / "Results"
                    / language
                    / "variance_adaptor_plots"
                    / kind
                    / f"{test_id}.png"
                )

                def validate_plot(p=path):
                    with Image.open(p) as image:
                        image.verify()
                    prediction = json.loads(
                        (
                            ROOT
                            / "analysis"
                            / language.lower()
                            / "predictions"
                            / f"{test_id}.json"
                        ).read_text()
                    )
                    if not prediction["checkpoint_step"] or not prediction["units"]:
                        raise ValueError("Plot has no trained-model provenance")
                    return "PNG decodes; corresponding predictions have checkpoint provenance"

                check(f"{test_id} {kind} plot", validate_plot)
        for filename in ["training_loss.png", "validation_loss.png"]:

            def image_check(p=ROOT / "Results" / language / filename):
                with Image.open(p) as image:
                    image.verify()
                return "PNG verified"

            check(f"{language} {filename}", image_check)

        def checkpoint_check(name=language):
            config = load_config(ROOT / name / "config.yaml")
            path = inference_checkpoint(config)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            complete = json.loads(
                (
                    ROOT / "analysis" / name.lower() / "training_complete.json"
                ).read_text()
            )
            config = load_config(ROOT / name / "config.yaml")
            if complete["step"] < config["training"]["min_steps"] or ckpt["step"] <= 0:
                raise ValueError("Full training has not reached the configured minimum")
            if not all(torch.isfinite(t).all() for t in ckpt["model"].values()):
                raise ValueError("Non-finite checkpoint values")
            from shared.model import build_model

            model, _ = build_model(ckpt["config"], ckpt["metadata"])
            model.load_state_dict(ckpt["model"])
            digest = sha256(path)
            for prediction_path in (
                ROOT / "analysis" / name.lower() / "predictions"
            ).glob("*.json"):
                if (
                    json.loads(prediction_path.read_text())["checkpoint_sha256"]
                    != digest
                ):
                    raise ValueError(
                        "Generated outputs came from a different checkpoint"
                    )
            return dict(
                step=ckpt["step"],
                best_validation=ckpt["best_validation"],
                parameters=sum(p.numel() for p in model.parameters()),
            )

        check(f"{language} checkpoint", checkpoint_check)

    def pdf_check():
        path = ROOT / "Report/Technical_Report.pdf"
        if path.read_bytes()[:5] != b"%PDF-":
            raise ValueError("Invalid PDF signature")
        from pypdf import PdfReader

        reader = PdfReader(path)
        text = "\n".join(page.extract_text() for page in reader.pages)
        for test_id in ["E1", "E2a", "E2b", "E3", "E4", "T1", "T2a", "T2b", "T3", "T4"]:
            if test_id not in text:
                raise ValueError(f"Report omits {test_id}")
        return dict(bytes=path.stat().st_size, pages=len(reader.pages))

    check("Technical report", pdf_check)
    check("README", lambda: dict(bytes=(ROOT / "README.md").stat().st_size))
    write_json(ROOT / "analysis/artifact_audit.json", checks)
    print(json.dumps(checks, indent=2))
    return all(r["status"] == "verified" for r in checks)


if __name__ == "__main__":
    sys.exit(0 if audit() else 1)
