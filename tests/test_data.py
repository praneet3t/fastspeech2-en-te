import pytest
from shared.data import select, transcript_key
from shared.io import ROOT, read_jsonl


def toy_rows():
    return [
        dict(
            source_id=str(i),
            audio_sha256=str(i),
            language="telugu",
            raw_text=f"text {i//2}",
            speaker_id=["a", "b"][i % 2],
            duration_seconds=10.0,
            split="unused",
            rejection_reason=None,
        )
        for i in range(100)
    ]


def test_deterministic_selection_and_group_disjointness():
    a = select(toy_rows(), 2026, train_seconds=500, val_seconds=100)
    b = select(toy_rows(), 2026, train_seconds=500, val_seconds=100)
    assert a == b
    train = [r for r in a if r["split"] == "train"]
    val = [r for r in a if r["split"] == "val"]
    assert not {r["source_id"] for r in train} & {r["source_id"] for r in val}
    assert not {transcript_key(r["raw_text"]) for r in train} & {
        transcript_key(r["raw_text"]) for r in val
    }
    assert {r["speaker_id"] for r in val} <= {r["speaker_id"] for r in train}
    assert sum(r["duration_seconds"] for r in train) == 500
    assert sum(r["duration_seconds"] for r in val) == 100


def test_insufficient_hours_explicit_failure():
    with pytest.raises(ValueError, match="eligible hours"):
        select(toy_rows(), 2026)


@pytest.mark.parametrize("language", ["english", "telugu"])
def test_actual_manifests(language):
    path = ROOT / "data/manifests" / f"{language}_train.jsonl"
    if not path.exists():
        pytest.skip(f"{language} real selection not created yet")
    train = read_jsonl(path)
    val = read_jsonl(path.with_name(f"{language}_val.jsonl"))
    for rows, split, target in [(train, "train", 18000), (val, "val", 3600)]:
        assert abs(sum(r["duration_seconds"] for r in rows) - target) < 0.05 * target
        for row in rows:
            assert row["language"] == language and row["split"] == split
            assert row["source_id"] and row["speaker_id"]
            assert row["sample_rate"] > 0 and row["channels"] >= 1
            assert (
                abs(row["sample_count"] / row["sample_rate"] - row["duration_seconds"])
                < 1e-8
            )
    assert not {r["audio_sha256"] for r in train} & {r["audio_sha256"] for r in val}
    assert not {transcript_key(r["raw_text"]) for r in train} & {
        transcript_key(r["raw_text"]) for r in val
    }


def test_split_skeleton_does_not_define_telugu_model_text():
    from shared.data import split_group_key
    from shared.text import frontend

    assert transcript_key("కల") != transcript_key("కిల")
    assert split_group_key("కల") == split_group_key("కిల")
    assert frontend("కల", "telugu").units != frontend("కిల", "telugu").units
