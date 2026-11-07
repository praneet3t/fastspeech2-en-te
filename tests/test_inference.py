import json
import numpy as np
import pytest
import soundfile as sf
from shared.inference import check_audio, inference_units, word_durations
from shared.text import TextInput, frontend
from shared.io import ROOT


def test_audio_validation(tmp_path):
    path = tmp_path / "sample.wav"
    samples = 0.1 * np.sin(2 * np.pi * 220 * np.arange(22050) / 22050)
    sf.write(path, samples, 22050)
    assert check_audio(path)["duration_seconds"] == 1
    sf.write(path, np.zeros(1000), 22050)
    with pytest.raises(ValueError, match="silent"):
        check_audio(path)
    sf.write(path, np.ones(1000), 22050)
    with pytest.raises(ValueError, match="clipping"):
        check_audio(path)
    sf.write(path, samples, 16000)
    with pytest.raises(ValueError, match="Invalid"):
        check_audio(path)


def test_boundary_silences_are_not_duplicated():
    units = inference_units(frontend("Hello.", "english"))
    assert units[0] == units[-1] == "sil"
    assert not any(a == b == "sil" for a, b in zip(units, units[1:]))


def test_vocoder_module_imports_without_model_import_order():
    import subprocess
    import sys
    from shared.io import ROOT

    subprocess.run(
        [sys.executable, "-c", "from shared.vocoder import load_vocoder"],
        cwd=ROOT,
        check=True,
    )


def test_asr_probe_word_errors_preserve_telugu_marks():
    from scripts.compare_checkpoints import words, edit_distance

    assert edit_distance(["a", "b", "c"], ["a", "x", "c", "d"]) == 2
    assert words("పట్టుచీరను.", "telugu") == ["పట్టుచీరను"]
    assert words("కల", "telugu") != words("కిల", "telugu")


@pytest.mark.parametrize("field", ["audio", "model"])
def test_incompatible_checkpoint_fails_before_model_or_vocoder_loading(tmp_path, field):
    import torch
    from shared.inference import synthesize

    config = dict(
        language="english", audio=dict(sample_rate=22050), model=dict(hidden=256)
    )
    saved = dict(config)
    saved[field] = dict(incompatible=True)
    checkpoint = tmp_path / "incompatible.pt"
    torch.save(dict(config=saved), checkpoint)
    with pytest.raises(ValueError, match=f"Checkpoint {field} differs"):
        synthesize(config, tmp_path / "unused.json", checkpoint)


def test_word_duration_exports_match_recorded_predictions():
    for record in json.loads((ROOT / "data/test_sentences.json").read_text()):
        prediction = json.loads(
            (
                ROOT
                / "analysis"
                / record["language"]
                / "predictions"
                / f'{record["test_id"]}.json'
            ).read_text()
        )
        text = frontend(record["original_text"], record["language"])
        assert (
            word_durations(
                text, prediction["units"], np.asarray(prediction["duration_frames"])
            )
            == prediction["word_predictions"]
        )


def test_word_duration_tracing_rejects_a_mismatched_sequence():
    text = TextInput("a", "a", ["a"], [["AH0"]], [])
    with pytest.raises(ValueError, match="Word/unit tracing mismatch"):
        word_durations(text, ["sil", "wrong", "sil"], np.array([1, 2, 1]))
