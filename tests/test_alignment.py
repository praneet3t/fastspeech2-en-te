import numpy as np
import pytest
from praatio import textgrid
from shared.alignment import read_alignment, resample


def grid_at(path, entries):
    grid = textgrid.Textgrid()
    grid.addTier(textgrid.IntervalTier("phones", entries, minT=0, maxT=1.0))
    grid.save(str(path), format="long_textgrid", includeBlankSpaces=True)


def test_shared_boundary_rounding_and_coverage(tmp_path):
    path = tmp_path / "example.TextGrid"
    grid_at(path, [(0.1, 0.35, "te_k"), (0.35, 0.8, "te_a")])
    units, durations, audit = read_alignment(path, ["te_k", "te_a"], 22050, 22050, 256)
    assert units == ["sil", "te_k", "te_a", "sil"]
    assert len(units) == len(durations) and durations.sum() == 22050 // 256
    assert (durations >= 0).all() and audit["silence_frames"] > 0


def test_missing_alignment(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing forced alignment"):
        read_alignment(tmp_path / "none", [], 22050, 22050, 256)


def test_wrong_phones_rejected(tmp_path):
    path = tmp_path / "bad.TextGrid"
    grid_at(path, [(0, 1, "te_a")])
    with pytest.raises(ValueError, match="sequence mismatch"):
        read_alignment(path, ["te_b"], 22050, 22050, 256)


def test_resampling_changes_sample_count():
    y = np.sin(2 * np.pi * 200 * np.arange(48000) / 48000).astype("float32")
    out = resample(y, 48000, 22050)
    assert len(out) == 22050 and abs(np.max(out) - 1) < 0.02
