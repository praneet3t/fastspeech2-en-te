"""Render the submission report from saved measurements and predictions."""

import argparse
import html
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shared.io import ROOT, read_jsonl


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def esc(value):
    return html.escape(str(value))


def paragraph(text):
    return f"<p>{text}</p>"


def table(headers, rows):
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{esc(x)}</th>" for x in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{esc(x)}</td>" for x in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table>"
    )


def image(path, caption, height=230):
    path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return f'<figure><img style="max-height:{height}px" src="{path.as_uri()}"><figcaption>{esc(caption)}</figcaption></figure>'


def code(text):
    return "<pre>" + esc(text) + "</pre>"


def compact_variances(prediction, output):
    fig, axes = plt.subplots(3, 1, figsize=(9, 3.3), sharex=True)
    for ax, key, label in zip(
        axes,
        ["duration_ms", "pitch_continuous_hz", "energy_stft_l2"],
        ["Duration\n(ms)", "Continuous F0\n(Hz)", "Energy\n(STFT L2)"],
    ):
        values = prediction[key]
        x = np.arange(1, len(values) + 1)
        if key == "duration_ms":
            ax.bar(x, values, color="#315c7c", width=0.8)
        else:
            ax.plot(x, values, color="#315c7c", linewidth=1)
        ax.set_ylabel(
            label, rotation=0, ha="right", va="center", fontsize=9, labelpad=10
        )
        ax.tick_params(labelsize=9)
        ax.grid(axis="y", alpha=0.15)
    axes[-1].set_xlabel("Linguistic unit index", fontsize=10)
    fig.tight_layout(pad=0.8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def build():
    manifest = load("data/test_sentences.json")
    source = load("analysis/assignment_source.json")
    languages = ["english", "telugu"]
    preprocessing = {
        name: load(f"analysis/{name}_preprocessing.json") for name in languages
    }
    selection = {
        name: load(f"analysis/{name}/checkpoint_selection.json") for name in languages
    }
    evaluation = {name: load(f"analysis/{name}/evaluation.json") for name in languages}
    metrics = {
        name: read_jsonl(ROOT / f"analysis/{name}/metrics.jsonl") for name in languages
    }
    figures = ROOT / "artifacts/report_figures"
    figures.mkdir(parents=True, exist_ok=True)
    predictions = {}
    for row in manifest:
        test_id = row["test_id"]
        prediction = load(f"analysis/{row['language']}/predictions/{test_id}.json")
        if (
            row["original_text"] != source["sentences"][test_id]
            or prediction["original_text"] != row["original_text"]
        ):
            raise ValueError("Report inputs do not match the verified assignment")
        predictions[test_id] = prediction
        compact_variances(prediction, figures / f"{test_id}.png")

    fig, axes = plt.subplots(1, 2, figsize=(9, 2.55))
    for ax, language in zip(axes, languages):
        rows = [r for r in metrics[language] if r["split"] == "val"]
        picked = selection[language]
        ax.plot([r["step"] for r in rows], [r["total"] for r in rows], color="#315c7c")
        ax.axvline(
            picked["selected_step"], color="#333333", linestyle="--", label="Selected"
        )
        ax.scatter(
            [picked["minimum_objective_step"]],
            [picked["minimum_objective_total"]],
            color="#555555",
            s=25,
            label="Loss minimum",
        )
        ax.set(
            title=language.title(), xlabel="Optimizer step", ylabel="Validation total"
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(figures / "checkpoint_selection.png", dpi=180)
    plt.close(fig)

    title = "Implementation of FastSpeech 2-Based Text-to-Speech Synthesis"
    pages = []

    def page(heading, content, first=False):
        display_title = title.replace(" Text-to-Speech", "<br>Text-to-Speech")
        report_title = f"<h1 class='report-title'>{display_title}</h1>" if first else ""
        pages.append(
            f"<section class='page'>{report_title}<h1>{heading}</h1>{content}<footer>{len(pages)+1}</footer></section>"
        )

    page(
        "1. Objective and implementation",
        paragraph(
            "I built separate English and Telugu text-to-speech systems around FastSpeech 2. Each acoustic model was trained on approximately five hours of speech, with about one hour reserved for validation. A fixed universal HiFi-GAN vocoder produces the waveform. The ten assignment sentences serve as test cases for examining duration, pitch and energy."
        )
        + image(
            "references/architecture.svg",
            "Model architecture and the supervised targets used during training.",
            365,
        )
        + paragraph(
            "The implementation starts from ming024/FastSpeech2. Its FFT encoder and decoder, scalar-F0 variance adaptor, five-part loss and PostNet are retained; the main adaptations concern data preparation, text processing, alignment and training. This is the scalar-pitch variant, rather than the later CWT representation or FastSpeech 2s."
        )
        + table(
            ["Reused", "Changed for this project"],
            [
                [
                    "FFT blocks, attention and PostNet",
                    "Language vocabularies and dataset preparation",
                ],
                [
                    "Variance adaptor and speaker embedding",
                    "English/Telugu text processing and forced alignment",
                ],
                [
                    "Duration expansion and loss functions",
                    "Feature extraction, training driver and analysis exports",
                ],
                [
                    "Fixed universal HiFi-GAN",
                    "Matching acoustic conditioning and waveform checks",
                ],
            ],
        ),
        first=True,
    )

    rows = []
    for language, dataset in [("english", "Svarah"), ("telugu", "IndicTTS Telugu")]:
        item = preprocessing[language]
        rows.append(
            [
                f"{language.title()} / {dataset}",
                f"{item['splits']['train']['utterances']:,} / {item['splits']['val']['utterances']:,}",
                f"{item['splits']['train']['hours']:.6f} / {item['splits']['val']['hours']:.6f}",
                f"{item['rejected']} / {item['selected']}",
            ]
        )
    page(
        "2. Datasets and text preparation",
        table(
            [
                "Dataset",
                "Accepted train / val utterances",
                "Accepted train / val hours",
                "Rejected / selected",
            ],
            rows,
        )
        + paragraph(
            "The datasets are pinned snapshots of AI4Bharat Svarah and SPRINGLab IndicTTS Telugu. Selection targeted 18,000 training seconds and 3,600 validation seconds per language, keeping whole utterances of at most 30 seconds with seed 2026. Alignment and feature checks reduced the accepted totals slightly."
        )
        + paragraph(
            "The source audit rejected invalid, near-silent, heavily clipped or duplicate audio and unsupported text. Matching transcripts and original-recording IDs were grouped before splitting. A seeded ordering and speaker-duration allocation kept each group within one split. The recorded checks found no exact train/validation overlap in IDs, audio hashes or normalized transcripts."
        )
        + table(
            ["", "English", "Telugu"],
            [
                ["Recordings", "Heterogeneous ASR speech", "Read-speech TTS corpus"],
                [
                    "Speaker information",
                    "Speaker IDs unavailable",
                    "Female and male source labels",
                ],
                [
                    "Model conditioning",
                    "Unconditioned speaker mixture",
                    "Learned speaker embedding",
                ],
                ["Submitted voice", "No selectable identity", "Female label"],
            ],
        )
        + paragraph(
            "English uses CMUdict pronunciations with g2p-en fallback. A context rule selects noun or verb stress for conduct. Dates and times are expanded before synthesis. The supplied rhyth m split and the word andaroo are retained."
        )
        + paragraph(
            "Telugu text is NFC-normalized and mapped through consonants, inherent vowels, vowel signs, virama and marks. These spelling-derived units are used in both alignment and inference. Number and time expansion uses explicit composition rules. The corrected assignment sentences match the original PDF; layout whitespace is normalized while visible word splits, including అడ విలో, remain."
        )
        + paragraph(
            "Configuration files pin the dataset revisions. Accepted totals are recorded in analysis/*_preprocessing.json, and data/manifests/selection_index.json preserves selected source IDs and audio hashes. Raw recordings and transcript-bearing manifests are not included in Git."
        ),
    )

    page(
        "3. Alignment and acoustic features",
        paragraph(
            "FastSpeech 2 needs a duration target for every input unit. Montreal Forced Aligner (MFA) produces those targets from audio, word labels and a pronunciation dictionary. Alignment uses 16 kHz audio. Opaque word IDs prevent MFA text cleaning from changing the Telugu dictionary spellings."
        )
        + paragraph(
            "English uses the english_us_arpa acoustic model. Telugu uses a separate HMM/GMM aligner trained only on its training split. Validation is aligned with fixed acoustic weights. The recorded Telugu validation export uses speaker-independent first-pass alignment. A small MFA 3.3.9 parser patch preserves labels such as te_aa instead of stripping them to te."
        )
        + paragraph(
            "Utterances are accepted only after checks on phone order, plausible durations and frame totals. These checks exclude 30 of 4,042 selected English utterances; missing alignments exclude 14 of 1,829 Telugu utterances. They catch structural problems, although the boundaries were not manually labelled."
        )
        + table(
            ["Feature", "Extraction"],
            [
                [
                    "Mel spectrogram",
                    "22,050 Hz; 80 Slaney magnitude bands; natural logarithm",
                ],
                [
                    "STFT",
                    "FFT/window 1,024; hop 256; reflect padding 384 per side; center=False",
                ],
                [
                    "Pitch",
                    "WORLD DIO and StoneMask; voiced F0 interpolated across unvoiced frames",
                ],
                [
                    "Unit targets",
                    "Pitch and spectral-L2 energy averaged over aligned units",
                ],
                [
                    "Normalization",
                    "Training data only; pitch statistics use voiced training frames",
                ],
            ],
        )
        + paragraph(
            "The mel recipe matches the fixed HiFi-GAN vocoder. A hop of 256 at 22,050 Hz gives about 11.61 ms per frame. Local NPZ files store tokens, aligned durations, standardized pitch/energy and mel targets."
        )
        + paragraph(
            "Continuous F0 in these plots includes interpolation through unvoiced regions; it is not a voicing prediction. Energy is a spectral L2 quantity, not decibels. The regression output is unbounded, so slightly negative inverse-standardized energy values, such as T3's minimum, are prediction errors rather than physically negative measurements."
        )
        + paragraph(
            "Implementation: shared/alignment.py, shared/features.py and shared/batches.py. The corpus-specific preprocessing replaces the upstream pipeline while retaining its scalar pitch approach."
        ),
    )

    page(
        "4. Model and training",
        paragraph(
            "Four FFT encoder blocks turn unit embeddings into contextual representations. Each block combines attention with convolutional feed-forward layers. Telugu adds a learned speaker vector before the variance adaptor; English has no speaker embedding because speaker identities are unavailable."
        )
        + paragraph(
            "The adaptor predicts duration before variance conditioning. Pitch embeddings are added before energy prediction, followed by energy embeddings. The length regulator repeats each conditioned vector for its duration in mel frames. Six FFT decoder blocks, a linear projection and a PostNet residual produce the final mel spectrogram."
        )
        + table(
            ["Setting", "English", "Telugu"],
            [
                ["Hidden width / attention heads", "256 / 2", "256 / 2"],
                ["Feed-forward width", "1,024", "1,024"],
                [
                    "Encoder / decoder / predictor dropout",
                    "0.2 / 0.2 / 0.5",
                    "0.2 / 0.2 / 0.5",
                ],
                ["Batch × gradient accumulation", "16 × 2", "8 × 4"],
                ["Nominal effective batch / precision", "32 / BF16", "32 / BF16"],
                ["Length-bucket multiplier", "1", "4"],
                ["Final step / selected step", "8,000 / 8,000", "14,500 / 10,500"],
            ],
        )
        + paragraph(
            "The objective is the equally weighted sum of five masked losses: mel L1, PostNet-mel L1, <span class='math'>log(1 + duration)</span> MSE, standardized pitch MSE and standardized energy MSE. Training supplies aligned duration, pitch and energy targets. Full inference predicts them."
        )
        + paragraph(
            "Both models were optimized with Adam: betas (0.9, 0.98), epsilon 1e-9, a 4,000-step warmup and gradient clipping at 1.0. Validation and checkpoints are recorded every 500 optimizer steps. Training requires at least 8,000 steps and stops when improvement plateaus over eight validation events, before the configured 30,000-step ceiling."
        )
        + paragraph(
            "Both runs used Linux, Python 3.10.12, PyTorch 2.5.1/CUDA 12.1 and an NVIDIA A100 40 GB. The saved runtimes are approximately 2,054 seconds for English and 6,188 seconds for Telugu. Checkpoints include vocabulary, feature statistics, speaker metadata and resume state. No training or model weights were changed for the report revision."
        ),
    )

    for language in languages:
        name = language.title()
        train = [r for r in metrics[language] if r["split"] == "train"]
        validation = [r for r in metrics[language] if r["split"] == "val"]
        if language == "english":
            interpretation = "English training loss keeps falling, but validation reaches its minimum at step 1,000 and then stays higher. The widening gap points to limited generalization. The later step-8,000 model was selected because it performed better on the fixed validation intelligibility probe, not because its reconstruction loss was lower."
        else:
            interpretation = "Telugu training and validation losses decrease before validation levels off. Step 10,500 has the lowest validation total; training continues to step 14,500 without a meaningful improvement under the stopping rule. Synthesis uses the earlier selected checkpoint."
        page(
            f"5. Training results: {name}",
            image(
                f"Results/{name}/training_loss.png",
                f"{name} training loss components.",
                300,
            )
            + image(
                f"Results/{name}/validation_loss.png",
                f"{name} validation loss components; target duration, pitch and energy are supplied.",
                300,
            )
            + paragraph(
                f"The logged training total falls from {train[0]['total']:.6f} at the first logged event to {train[-1]['total']:.6f} at the last. Validation changes from {validation[0]['total']:.6f} to {validation[-1]['total']:.6f}."
            )
            + paragraph(interpretation)
            + paragraph(
                "Training points average minibatches; validation uses corpus-weighted means. The component units differ, and the total is not a speech-quality score."
            ),
        )

    eval_rows = []
    for language in languages:
        item = evaluation[language]
        eval_rows.append(
            [
                language.title(),
                f"{item['total']:.6f}",
                f"{item['duration_mae_frames']:.6f}",
                f"{item['pitch_rmse_hz_mostly_voiced_units']:.6f}",
                f"{item['energy_rmse_stft_l2']:.6f}",
            ]
        )
    page(
        "6. Selection, evaluation and limitations",
        image(
            "artifacts/report_figures/checkpoint_selection.png",
            "Validation total, loss minimum and selected inference checkpoint.",
            205,
        )
        + paragraph(
            "English's lowest logged validation total is 3.582084 at step 1,000. On six fixed validation utterances, Whisper gives 34/129 word errors at that checkpoint and 10/129 at step 8,000. The probe therefore favors step 8,000 despite its higher logged validation total of 3.879625. Telugu selects step 10,500, the validation minimum of 1.977242. Assignment sentences were not used for selection."
        )
        + table(
            [
                "Selected model",
                "Saved evaluation total",
                "Duration MAE (frames)",
                "F0 RMSE (Hz)",
                "Energy RMSE (STFT L2)",
            ],
            eval_rows,
        )
        + paragraph(
            "These saved evaluations supply target variances to the mel path. F0 error uses nonzero-duration units with at least 80% voiced frames. English's separately recomputed total differs slightly from its logged value; both records are retained."
        )
        + paragraph(
            "Telugu has lower conditional errors, but its read speech, two speaker labels and different recordings/alignments make this an uneven comparison with Svarah. English has no selectable speaker identity. Telugu's spelling-derived units still need native pronunciation review."
        )
        + paragraph(
            "Small validation controls compare original speech, reference-mel vocoding, target-variance synthesis and full inference. Recognition, timing and reconstruction do not always agree. Unconstrained DTW can hide repetition, and the reused subset is too small to isolate a unique source of error."
        )
        + paragraph(
            "No MOS or qualified listening study was collected. English observations use saved ASR transcripts; corrected Telugu observations use current text and variance predictions. Robotic delivery, pronunciation, stress and voice consistency therefore remain listening concerns. Lower loss or a fluent transcript does not establish natural speech."
        ),
    )

    notes = {
        "E1": "Whisper splits antidisestablishmentarianism into 'entire establishment Aryanism' while retaining most of the sentence. This flags the long word as a weak point; the duration plot shows its allocation rather than resolving whether the error is pronunciation or recognition.",
        "E2a": "The frontend supplies noun stress for conduct, and Whisper reproduces the sentence. Its duration, pitch and energy pattern can be compared with E2b, but the recognizer does not measure how convincingly the stress contrast is spoken.",
        "E2b": "The verb pronunciation of conduct differs from E2a in the input stress pattern. Whisper reproduces this sentence too. The contrast is explicitly supplied by text processing rather than learned from raw spelling alone.",
        "E3": "The input keeps rhyth and the letter-name m separate, while Whisper returns ordinary 'rhythm'. This is a useful failure case: recognition can smooth over the unusual word boundary even though the model receives separate units.",
        "E4": "The normalizer expands the date and 10:45 AM, including 'ay em'. Whisper nevertheless returns 'p.m.' and corrupts the andaroo/should-arrive span. Correct symbolic expansion does not by itself solve difficult-word or acoustic errors.",
        "T1": "The technical word న్యూరోఫార్మకోలాజికల్ is kept as written in the assignment. Its unit allocations and changing F0/energy show how the model handles the long span. The continuous F0 trace should be read alongside the long final pause, not as evidence of voicing throughout.",
        "T2a": "పట్టు and చాలా are separate words in the input. Their boundaries can be traced individually, though the frontend flattens the same unit sequence used for synthesis. Meaning is not an explicit model input; compare the timing and pitch pattern with the silk-cloth context in T2b.",
        "T2b": "This sentence uses పట్టు in the silk-cloth context. The input preserves the boundary before చీరను and the spelling వచ్చింది. The large trailing duration allocation is silence-labelled, so it represents an end pause rather than a stretched final speech sound.",
        "T3": "The assignment's అడ విలో split is retained. The duration plot is dominated by the final silence unit, while speech units are much shorter. The small negative energy prediction is an unconstrained regression error; it is not a negative acoustic measurement.",
        "T4": "The year, ordinal date and time are expanded before inference, and the closing word is సిద్ధంగా. This longer sentence contains several silence-labelled allocations. The plots show timing and variance variation across the expanded words; spoken-number accuracy still needs listening.",
    }

    def test_block(row):
        test_id = row["test_id"]
        item = predictions[test_id]
        pitch, energy = item["pitch_continuous_hz"], item["energy_stft_l2"]
        end_pause = item["duration_ms"][-1]
        summary = f"{item['audio']['duration_seconds']:.3f} s; {len(item['units'])} units; final silence {end_pause:.1f} ms. F0 {min(pitch):.1f}-{max(pitch):.1f} Hz; energy {min(energy):.2f}-{max(energy):.2f} STFT L2."
        return (
            f"<article><h2>{test_id} · {row['language'].title()}</h2><p class='sentence' lang='{'te' if row['language'] == 'telugu' else 'en'}'>{esc(row['original_text'])}</p>"
            + image(f"artifacts/report_figures/{test_id}.png", "", 245)
            + f'<p class="statistics">{esc(summary)}</p>'
            + paragraph(esc(notes[test_id]))
            + "</article>"
        )

    for start in range(0, len(manifest), 2):
        pair = manifest[start : start + 2]
        introduction = (
            paragraph(
                "Plots show predicted duration, continuous F0 and spectral energy by unit index. Recordings and full-resolution plots are in Results/, named by test ID."
            )
            if start == 0
            else ""
        )
        page(
            f"7. Test sentences: {pair[0]['test_id']} and {pair[1]['test_id']}",
            introduction + "".join(test_block(row) for row in pair),
        )

    commands = """bash scripts/setup.sh
source .venv/bin/activate
python scripts/download_artifacts.py
mkdir -p artifacts
cp data/test_sentences.json artifacts/tests.json
python English/inference.py \\
  --test-manifest artifacts/tests.json \\
  --output-root artifacts/inference
python Telugu/inference.py \\
  --test-manifest artifacts/tests.json \\
  --output-root artifacts/inference"""
    training = """# Separate checkout: move existing metrics logs aside first.
hf auth login
python scripts/download_data.py english
python scripts/download_data.py telugu
python English/preprocess.py
python Telugu/preprocess.py --train-aligner
python English/train.py
python Telugu/train.py"""
    page(
        "8. Reproduction and references",
        paragraph(
            "Use the language configs and pinned Python requirements. The original training environment was Linux/Python 3.10/PyTorch 2.5.1/CUDA 12.1. Install ffmpeg, libsndfile and Telugu fonts; preprocessing also requires a separate MFA 3.3.9 environment and the supplied label-parser patch. The README provides the full setup."
        )
        + code(commands)
        + paragraph(
            "The selected acoustic checkpoints are provided in the repository's models-v1 release. The downloader checks sizes and hashes against the artifact manifest. HiFi-GAN is obtained from its original upstream distribution, rather than rehosted here. Checkpoints embed the vocabulary, normalization statistics and speaker map, so inference does not require training recordings."
        )
        + code(training)
        + paragraph(
            "The current Telugu recordings were generated on CPU with Python 3.12/PyTorch 2.12.1. Original English recordings are retained. Different runtimes can produce different waveforms even with matching units and durations. To rebuild the report, run python scripts/make_report.py; it reads saved measurements and does not run experiments."
        )
        + "<h2>References and attribution</h2>"
        + paragraph(
            "[1] Ren et al. <i>FastSpeech 2: Fast and High-Quality End-to-End Text to Speech.</i> <a href='https://arxiv.org/abs/2006.04558'>arXiv:2006.04558</a>.<br>[2] <a href='https://github.com/ming024/FastSpeech2'>ming024/FastSpeech2</a>; retained scalar-F0 implementation and HiFi-GAN source.<br>[3] AI4Bharat / Javed et al. <a href='https://huggingface.co/datasets/ai4bharat/Svarah'>Svarah</a>, Interspeech 2023.<br>[4] Speech Technology Consortium, IIT Madras. <a href='https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu'>IndicTTS Telugu</a> (SPRINGLab mirror).<br>[5] Kong et al. <i>HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis.</i> NeurIPS 2020."
        )
        + paragraph(
            "Both vendored MIT licenses and NOTICE are retained. Svarah uses CC BY 4.0 and provider access conditions. IndicTTS uses its custom provider EULA, included under references/licenses/. Raw corpora are not redistributed. The source-code license for HiFi-GAN does not independently establish public redistribution rights for its pretrained weights."
        ),
    )

    css = """
@page { size: A4; margin: 0; }
* { box-sizing: border-box; }
body {
    margin: 0;
    color: #202a32;
    font: 10.5pt/1.42 Arial, 'Nirmala UI', 'Noto Sans Telugu', sans-serif;
    text-rendering: optimizeLegibility;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
}
.page {
    position: relative;
    width: 210mm;
    height: 297mm;
    padding: 18mm 20mm 20mm;
    break-after: page;
    overflow: hidden;
}
.page:last-child { break-after: auto; }
h1 {
    margin: 0 0 4mm;
    color: #203f55;
    font-size: 16pt;
    line-height: 1.2;
    break-after: avoid;
}
.report-title {
    margin-bottom: 5mm;
    color: #202a32;
    font-size: 18pt;
    line-height: 1.22;
}
h2 {
    margin: 4mm 0 2mm;
    font-size: 11.8pt;
    line-height: 1.25;
    break-after: avoid;
}
p { margin: 2.6mm 0; orphans: 3; widows: 3; }
figure { margin: 3mm 0; break-inside: avoid; }
img { display: block; max-width: 100%; margin: auto; object-fit: contain; }
figcaption { margin-top: 1mm; color: #52616b; font-size: 8.5pt; line-height: 1.35; }
figcaption:empty { display: none; }
table {
    width: 100%;
    margin: 3.5mm 0;
    border-collapse: collapse;
    font-size: 9.4pt;
    line-height: 1.35;
    break-inside: avoid;
}
th, td {
    padding: 2.2mm 2mm;
    border-bottom: 0.5pt solid #d2d9dd;
    text-align: left;
    vertical-align: top;
    font-variant-numeric: tabular-nums;
}
th { background: #edf1f4; font-weight: 700; }
pre {
    max-width: 100%;
    margin: 3mm 0;
    padding: 3mm;
    background: #f2f4f5;
    font: 8.5pt/1.42 Consolas, monospace;
    white-space: pre;
    break-inside: avoid;
}
a { color: #244f70; text-decoration: none; }
footer {
    position: absolute;
    right: 20mm;
    bottom: 10mm;
    left: 20mm;
    color: #52616b;
    font-size: 9pt;
    text-align: center;
}
.sentence { font-size: 11pt; line-height: 1.6; }
.statistics { font-size: 9.5pt; line-height: 1.4; font-variant-numeric: tabular-nums; }
.math { white-space: nowrap; }
article { margin: 0 0 6mm; break-inside: avoid; }
article h2 { margin-top: 3mm; }
"""
    destination = ROOT / "Report/Technical_Report.html"
    destination.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>'
        + title
        + "</title><style>"
        + css
        + "</style><body>"
        + "".join(pages)
        + "</body></html>",
        encoding="utf-8",
    )
    return destination, len(pages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", help="Path to a Chromium/Chrome/Edge executable")
    args = parser.parse_args()
    source, pages = build()
    output = ROOT / "Report/Technical_Report.pdf"
    candidates = [
        args.browser,
        shutil.which("chromium"),
        shutil.which("google-chrome"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]
    browser = next((p for p in candidates if p and Path(p).is_file()), None)
    if browser:
        profile = ROOT / "artifacts/report_browser_profile"
        subprocess.run(
            [
                browser,
                "--headless",
                "--disable-gpu",
                "--no-sandbox",
                "--no-pdf-header-footer",
                "--allow-file-access-from-files",
                f"--user-data-dir={profile}",
                f"--print-to-pdf={output}",
                source.as_uri(),
            ],
            check=True,
            timeout=120,
            capture_output=True,
        )
    else:
        from weasyprint import HTML

        HTML(filename=str(source)).write_pdf(str(output))
    if not output.is_file() or output.stat().st_size < 10000:
        raise RuntimeError("Report renderer did not produce a PDF")
    print(f"Rendered {pages} designed pages: {output}")


if __name__ == "__main__":
    main()
