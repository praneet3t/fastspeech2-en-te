"""Build inference inputs from the verified assignment transcription."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.io import ROOT, write_json
from shared.text import frontend
from shared.inference import inference_units

source = json.loads(
    (ROOT / "analysis/assignment_source.json").read_text(encoding="utf-8")
)
sentences = source["sentences"]
existing = {
    row["test_id"]: row
    for row in json.loads(
        (ROOT / "data/test_sentences.json").read_text(encoding="utf-8")
    )
}
records = []
for test_id, original in sentences.items():
    language = "english" if test_id.startswith("E") else "telugu"
    text = frontend(original, language)
    row = dict(
        test_id=test_id,
        language=language,
        original_text=original,
        internal_normalized_text=text.normalized,
        model_input_sequence=inference_units(text),
        normalization_log=text.transformations,
        speaker_id=None,
        model_checkpoint=None,
        vocoder_checkpoint="artifacts/vocoder/generator_universal.pth.tar",
        output_audio_filename=f"Results/{language.title()}/test_generated_audios/{test_id}.wav",
        text_source="FastSpeech2_Assignment.pdf; visually verified 2026-11-07; extraction and whitespace policy in analysis/assignment_source.json",
        pdf_visually_verified=True,
        source_pdf_sha256=source["source_sha256"],
    )
    if original == existing[test_id]["original_text"]:
        row = {
            **existing[test_id],
            **{
                k: row[k]
                for k in ["text_source", "pdf_visually_verified", "source_pdf_sha256"]
            },
        }
    records.append(row)
write_json(ROOT / "data/test_sentences.json", records)
