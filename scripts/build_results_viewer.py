"""Build one local audio-and-plots viewer using relative file paths."""

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    rows = json.loads((ROOT / "data/test_sentences.json").read_text(encoding="utf-8"))
    notes = json.loads(
        (ROOT / "analysis/test_observations.json").read_text(encoding="utf-8")
    )
    body = [
        "<h1>FastSpeech 2: audio and predictions</h1>",
        "<p>English selected step 8,000; Telugu selected step 10,500. Telugu inputs were verified against the assignment PDF and regenerated on 7 November 2026 with the original weights. No qualified listening study was conducted.</p>",
        '<p><a href="Report/Technical_Report.pdf">Technical report</a> · <a href="README.md">Project guide</a></p>',
    ]
    for row in rows:
        key = row["test_id"]
        language = row["language"]
        p = json.loads(
            (ROOT / f"analysis/{language}/predictions/{key}.json").read_text(
                encoding="utf-8"
            )
        )
        body.append(
            f'<article id="{key}"><h2>{key}</h2><p class="sentence">{html.escape(row["original_text"])}</p>'
        )
        body.append(
            f'<audio controls preload="none" data-language="{language}" data-test="{key}" src="{row["output_audio_filename"]}"></audio>'
        )
        body.append(
            f'<p>{p["audio"]["duration_seconds"]:.3f} s · 22,050 Hz mono · {html.escape(p["speaker_id"])}.</p>'
        )
        body.append(
            "<p>"
            + html.escape(notes[key]["observation"])
            + "</p><details><summary>Duration, pitch and energy predictions</summary>"
        )
        for kind in ["duration", "pitch", "energy"]:
            path = f"Results/{language.title()}/variance_adaptor_plots/{kind}/{key}.png"
            body.append(
                f'<figure><img loading="lazy" src="{path}" alt="{key} predicted {kind}"><figcaption>{kind.title()}: model prediction per linguistic unit.</figcaption></figure>'
            )
        body.append("</details></article>")
    body.append("<h2>Training and validation curves</h2>")
    for language in ["English", "Telugu"]:
        for split in ["training", "validation"]:
            path = f"Results/{language}/{split}_loss.png"
            body.append(
                f'<figure><img loading="lazy" src="{path}" alt="{language} {split} loss"><figcaption>{language} {split}; validation is teacher-forced.</figcaption></figure>'
            )
    css = "body{max-width:940px;margin:32px auto;padding:0 20px;font:16px/1.5 system-ui,'Nirmala UI','Noto Sans Telugu',sans-serif;color:#24313b}h1,h2{color:#254961}article{padding:16px 0;border-bottom:1px solid #ccd3d8}.sentence{font-size:18px}audio,img{width:100%}figure{margin:16px 0}figcaption{font-size:13px;color:#50616c}summary{cursor:pointer}a{color:#254961}"
    output = ROOT / "results.html"
    output.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FastSpeech 2 results</title><style>'
        + css
        + "</style><body>"
        + "".join(body)
        + "</body></html>",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    build()
