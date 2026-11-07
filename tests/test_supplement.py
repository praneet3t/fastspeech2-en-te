"""Check diagnostic conventions and the executed supplement's actual exports."""

import json

import numpy as np
from scripts.research_supplement import coarse_dtw
from shared.io import ROOT


def load(name):
    return json.loads((ROOT / "analysis/supplement" / name).read_text())


def test_dtw_can_hide_repetition_and_is_not_a_timing_score():
    reference = np.array([[0.0] * 80] * 4 + [[1.0] * 80] * 4)
    repeated = np.repeat(reference, 2, axis=0)
    assert coarse_dtw(reference, reference) == 0
    assert coarse_dtw(reference, repeated) == 0
    assert len(repeated) / len(reference) == 2
    assert coarse_dtw(reference, reference + 2) == 2


def test_executed_matched_checkpoint_inputs_and_duration_conventions():
    metrics = load("metrics.json")
    predictions = load("variance_predictions.json")
    assert len(metrics) == 44 and len(predictions) == 14
    for case in [f"e{i}" for i in range(1, 7)]:
        pair = [p for p in predictions if p["case_id"] == case]
        assert {p["checkpoint"] for p in pair} == {"step1000", "step8000"}
        assert pair[0]["aligned_units"] == pair[1]["aligned_units"]
        assert pair[0]["full_inference_units"] == pair[1]["full_inference_units"]
        assert pair[0]["targets"] == pair[1]["targets"]
    for p in predictions:
        assert [u for u in p["aligned_units"] if u != "sil"] == [
            u for u in p["full_inference_units"] if u != "sil"
        ]
        for mode, key in [("gt_variance", "targets"), ("full_inference", None)]:
            row = next(
                r
                for r in metrics
                if r["case_id"] == p["case_id"]
                and r["checkpoint"] == p["checkpoint"]
                and r["mode"] == mode
            )
            durations = p[key]["duration"] if key else p["full_duration_frames"]
            assert sum(durations) * 256 == row["audio"]["samples"]
            if mode == "full_inference":
                assert "frame_logmel_l1" not in row["acoustic"]


def test_local_mel_arrays_behind_the_four_comparison_figures():
    import pytest

    folder = ROOT / "artifacts/research_supplement"
    if not (folder / "english_e1_mels.npz").exists():
        pytest.skip(
            "Reference-derived mel arrays stay local; execute the supplement to regenerate"
        )
    for name in ["english_e1", "english_e2", "telugu_t1", "telugu_t2"]:
        with np.load(folder / f"{name}_mels.npz") as arrays:
            assert len(arrays.files) == (6 if name.startswith("english") else 4)
            assert "Reference mel" in arrays.files
            for key in arrays.files:
                values = arrays[key]
                assert values.ndim == 2 and values.shape[1] == 80
                assert np.isfinite(values).all()
