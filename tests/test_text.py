import json
import pytest
from shared.text import frontend, telugu_word
from shared.io import ROOT


def test_english_consistent_and_raw_preserved():
    original = "The rhyth m of the orchestra changed unexpectedly."
    a = frontend(original, "english")
    b = frontend(original, "english")
    assert a.original == original and "rhyth m" in a.normalized
    assert a.units == b.units and "rhyth" in a.words and "m" in a.words


def test_heteronym_context():
    noun = frontend("The orchestra's conduct was praised.", "english")
    verb = frontend("They will conduct an experiment.", "english")
    assert "conduct_noun" in noun.words and "conduct_verb" in verb.words
    assert (
        noun.pronunciations[noun.words.index("conduct_noun")]
        != verb.pronunciations[verb.words.index("conduct_verb")]
    )


def test_numbers_date_time():
    text = frontend("10:45 AM on 12 November 2026", "english")
    assert "ten forty-five ay em" in text.normalized and not any(
        c.isdigit() for c in text.normalized
    )
    te = frontend("2026 నవంబర్ 12వ తేదీ 10:45", "telugu")
    assert not any(c.isdigit() for c in te.normalized)
    assert "రెండు వేల ఇరవై ఆరు" in te.normalized and "పన్నెండవ" in te.normalized


@pytest.mark.parametrize(
    "word,phones",
    [
        ("క", ["te_k", "te_a"]),
        ("కా", ["te_k", "te_aa"]),
        ("క్", ["te_k"]),
        ("అ", ["te_a"]),
        ("క్క", ["te_k", "te_k", "te_a"]),
        ("క్రి", ["te_k", "te_r", "te_i"]),
        ("కం", ["te_k", "te_a", "te_anusvara"]),
        ("కః", ["te_k", "te_a", "te_visarga"]),
        ("న్యూ", ["te_n", "te_y", "te_uu"]),
    ],
)
def test_telugu_units(word, phones):
    assert telugu_word(word) == phones


def test_unknown_explicit_failure():
    with pytest.raises(ValueError, match="Unsupported"):
        frontend("తెలుగు 🐈", "telugu")
    with pytest.raises(ValueError, match="Unsupported"):
        frontend("hello 🐈", "english")


def test_combining_space_preserved_and_disclosed():
    original = "వచ్చ ింది."
    result = frontend(original, "telugu")
    assert result.original == original
    assert any("Orphan" in message for message in result.transformations)
    assert "te_i" in result.units


def test_required_manifest():
    data = json.loads((ROOT / "data/test_sentences.json").read_text())
    assert [r["test_id"] for r in data] == [
        "E1",
        "E2a",
        "E2b",
        "E3",
        "E4",
        "T1",
        "T2a",
        "T2b",
        "T3",
        "T4",
    ]
    for row in data:
        text = frontend(row["original_text"], row["language"])
        assert text.normalized == row["internal_normalized_text"]
        assert row["pdf_visually_verified"]
    assert (
        data[3]["original_text"] == "The rhyth m of the orchestra changed unexpectedly."
    )


def test_repeated_whitespace_preserved():
    result = frontend("hello   world", "english")
    assert result.original == "hello   world" and result.normalized == "hello world"
    assert any("whitespace" in x for x in result.transformations)


def test_telugu_assignment_source_is_authoritative():
    rows = json.loads((ROOT / "data/test_sentences.json").read_text())
    t4 = next(r for r in rows if r["test_id"] == "T4")
    source = json.loads(
        (ROOT / "analysis/assignment_source.json").read_text(encoding="utf-8")
    )
    assert t4["original_text"] == source["sentences"]["T4"]
    assert "సిద్ధంగా" in t4["original_text"]
    assert "\n" not in t4["internal_normalized_text"]
    for row in rows:
        assert row["original_text"] == source["sentences"][row["test_id"]]
        assert row["source_pdf_sha256"] == source["source_sha256"]
        if row["language"] == "telugu":
            assert [f"U+{ord(c):04X}" for c in row["original_text"]] == source[
                "telugu_codepoints"
            ][row["test_id"]]


def test_meridiem_uses_letter_names_and_nonascii_is_not_silently_dropped():
    result = frontend("10:45 AM", "english")
    assert result.pronunciations[result.words.index("ay")] == ["EY1"]
    with pytest.raises(ValueError, match="Unsupported English"):
        frontend("café", "english")
    with pytest.raises(ValueError, match="Invalid English time"):
        frontend("10:99 AM", "english")
