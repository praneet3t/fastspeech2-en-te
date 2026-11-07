"""Fixed-checkpoint diagnostics; writes only supplement/ and ignored artifacts/."""

import argparse
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyworld
import regex
import soundfile as sf
import torch
from shared.alignment import lexical_id, resample, read_alignment
from shared.batches import Features, collate, to_device
from shared.features import acoustic_features
from shared.inference import inference_units, check_audio
from shared.io import ROOT, load_config, read_jsonl, write_json, sha256
from shared.model import build_model
from shared.text import frontend
from shared.vocoder import load_vocoder, vocode
from scripts.compare_checkpoints import words, edit_distance

OUT = ROOT / "analysis/supplement"
LOCAL = ROOT / "artifacts/research_supplement"


def protected_hashes():
    """Protect every existing assignment result, report, manifest and checkpoint."""
    files = list((ROOT / "Results").rglob("*"))
    files += [
        ROOT / "Report/Technical_Report.pdf",
        ROOT / "Report/Technical_Report.html",
        ROOT / "data/test_sentences.json",
    ]
    for name in ["English", "Telugu"]:
        files += list((ROOT / name / "checkpoints").glob("*.pt"))
        files += [ROOT / name / "config.yaml"]
    return {str(p.relative_to(ROOT)): sha256(p) for p in files if p.is_file()}


def coarse_dtw(reference, predicted):
    """Mean log-mel L1 along an unconstrained DTW path, after 4-frame pooling.

    This discounts timing differences and can hide repetition/deletion errors.
    It is a descriptive spectral discrepancy, not an intelligibility metric.
    """

    def pool(x):
        return np.stack([x[i : i + 4].mean(0) for i in range(0, len(x), 4)])

    a, b = pool(reference), pool(predicted)
    _, path = librosa.sequence.dtw(X=a.T, Y=b.T, metric="cityblock")
    return float(np.abs(a[path[:, 0]] - b[path[:, 1]]).mean())


def f0(samples):
    values, times = pyworld.dio(
        samples.astype("float64"),
        22050,
        f0_floor=50,
        f0_ceil=600,
        frame_period=256 / 22050 * 1000,
    )
    return pyworld.stonemask(samples.astype("float64"), values, times, 22050)


def wave_comparison(reference, waveform, aligned):
    mel, energy = acoustic_features(waveform)
    refmel, refenergy = acoustic_features(reference)
    a, b = f0(reference), f0(waveform)
    result = dict(
        duration_ratio=len(waveform) / len(reference),
        coarse_dtw_logmel_l1=coarse_dtw(refmel, mel),
        voiced_f0_median_hz=float(np.median(b[b > 0])) if (b > 0).any() else None,
    )
    if aligned:
        if len(reference) != len(waveform):
            raise ValueError("Aligned waveform comparison needs equal lengths")
        n = min(len(a), len(b))
        common = (a[:n] > 0) & (b[:n] > 0)
        result.update(
            common_voiced_frames=int(common.sum()),
            f0_rmse_common_voiced_hz=(
                float(np.sqrt(np.mean((a[:n][common] - b[:n][common]) ** 2)))
                if common.any()
                else None
            ),
            voiced_decision_agreement=float(((a[:n] > 0) == (b[:n] > 0)).mean()),
            frame_logmel_l1=float(np.abs(mel - refmel).mean()),
            frame_logenergy_mae=float(
                np.abs(np.log(energy + 1e-5) - np.log(refenergy + 1e-5)).mean()
            ),
        )
    return result


def waveform(path, samples):
    sf.write(path, samples, 22050, subtype="FLOAT")
    return check_audio(path)


def choose_rows():
    # Reuse the original six English validation IDs, without selecting new wins.
    probe = json.loads((ROOT / "analysis/english/checkpoint_probe.json").read_text())
    english = {
        r["source_id"]: r for r in read_jsonl(ROOT / "data/features/english/val.jsonl")
    }
    english = [
        english[r["source_id"]] for r in probe["candidates"]["best"]["utterances"]
    ]
    telugu = read_jsonl(ROOT / "data/features/telugu/val.jsonl")
    telugu = [
        next(
            r
            for r in telugu
            if r["speaker_id"] == speaker and 5 <= r["duration_seconds"] <= 8
        )
        for speaker in ["female", "male"]
    ]
    return {"english": english, "telugu": telugu}


def unicode_audit(rows):
    tests = json.loads((ROOT / "data/test_sentences.json").read_text())
    cases = [
        (r["test_id"], r["original_text"], "assignment")
        for r in tests
        if r["language"] == "telugu"
    ]
    cases += [
        (f"D{i:02d}", text, "diagnostic, not training data")
        for i, text in enumerate(
            [
                "క",
                "కా",
                "కి",
                "క్",
                "క్క",
                "కై",
                "కై",
                "వచ్చ ింది",
                "వచ్చింది",
                "అడ విలో",
                "అడవిలో",
                "ి",
                "2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు",
                "24:99",
                "₹100",
                "తెలుగు abc",
            ]
        )
    ]
    records = []
    for case_id, raw, origin in cases:
        row = dict(
            case_id=case_id,
            origin=origin,
            raw_text=raw,
            codepoints=[
                dict(
                    character=c,
                    code=f"U+{ord(c):04X}",
                    name=unicodedata.name(c, "UNKNOWN"),
                    category=unicodedata.category(c),
                    combining_class=unicodedata.combining(c),
                )
                for c in raw
            ],
            grapheme_clusters=regex.findall(r"\X", raw),
        )
        try:
            text = frontend(raw, "telugu")
            row.update(
                normalized_text=text.normalized,
                normalization_log=text.transformations,
                words=text.words,
                pronunciations=text.pronunciations,
                alignment_lexicon=[
                    dict(key=lexical_id(w), units=p)
                    for w, p in zip(text.words, text.pronunciations)
                    if p not in (["sp"], ["sil"])
                ],
                inference_units=inference_units(text),
                outcome="accepted; pronunciation not independently validated",
            )
        except ValueError as error:
            row.update(outcome="explicit failure", error=str(error))
        records.append(row)
    alignment = []
    for row in rows["telugu"]:
        path = (
            ROOT
            / "data/features/telugu/raw"
            / f'{row["source_id"].replace(":","_")}.npz'
        )
        with np.load(path) as feature:
            actual = feature["units"].tolist()
            durations = feature["duration"].tolist()
        expected = [
            u for p in row["pronunciations"] for u in p if u not in ["sil", "sp"]
        ]
        assert [u for u in actual if u != "sil"] == expected
        grid = (
            ROOT
            / "data/alignment/telugu/aligned_val"
            / row["speaker_id"]
            / (row["source_id"].replace(":", "_") + ".TextGrid")
        )
        samples, rate = sf.read(
            ROOT / row["audio_path"], dtype="float32", always_2d=True
        )
        samples = resample(samples.mean(1), rate, 22050)
        checked_units, checked_durations, _ = read_alignment(
            grid, expected, len(samples), 22050, 256
        )
        assert checked_units == actual and checked_durations.tolist() == durations
        alignment.append(
            dict(
                source_id=row["source_id"],
                speaker_id=row["speaker_id"],
                linguistic_units=actual,
                duration_frames=durations,
                frames=sum(durations),
                expected_non_silence_units_match=True,
                textgrid_reparsed_and_frame_targets_match=True,
                textgrid=str(
                    (
                        ROOT
                        / "data/alignment/telugu/aligned_val"
                        / row["speaker_id"]
                        / (row["source_id"].replace(":", "_") + ".TextGrid")
                    ).relative_to(ROOT)
                ),
            )
        )
    write_json(
        OUT / "telugu_frontend.json",
        dict(
            cases=records,
            accepted_validation_alignment_checks=alignment,
            caveat="Test/dictionary traces are representations, not forced alignments of test audio. No synthetic alignment was created.",
        ),
    )


def tensor_summary(value):
    if isinstance(value, torch.Tensor):
        x = value.detach()
        return dict(
            shape=list(x.shape),
            dtype=str(x.dtype),
            finite=bool(torch.isfinite(x).all()),
            minimum=float(x.min()) if x.numel() else None,
            maximum=float(x.max()) if x.numel() else None,
        )
    if isinstance(value, (list, tuple)):
        return [tensor_summary(x) for x in value]
    return (
        value
        if isinstance(value, (int, float, str)) or value is None
        else str(type(value))
    )


@torch.inference_mode()
def execution_trace(language, model, vocoder, metadata, checkpoint):
    test_id = "E2a" if language == "english" else "T2a"
    record = next(
        r
        for r in json.loads((ROOT / "data/test_sentences.json").read_text())
        if r["test_id"] == test_id
    )
    text = frontend(record["original_text"], language)
    units = inference_units(text)
    speaker = record["speaker_id"]
    ids = torch.tensor([[metadata["vocab"][u] for u in units]], device="cuda")
    lengths = torch.tensor([len(units)], device="cuda")
    events = []
    hooks = []
    actual = {}
    names = {
        "token_embedding": model.encoder.src_word_emb,
        "encoder": model.encoder,
        "duration_predictor": model.variance_adaptor.duration_predictor,
        "pitch_predictor": model.variance_adaptor.pitch_predictor,
        "energy_predictor": model.variance_adaptor.energy_predictor,
        "pitch_embedding": model.variance_adaptor.pitch_embedding,
        "energy_embedding": model.variance_adaptor.energy_embedding,
        "length_regulator": model.variance_adaptor.length_regulator,
        "decoder": model.decoder,
        "mel_projection": model.mel_linear,
        "postnet_residual": model.postnet,
        "vocoder": vocoder,
    }
    if model.speaker_emb is not None:
        names["speaker_embedding"] = model.speaker_emb

    def capture(name):
        def hook(module, args, output):
            events.append(
                dict(
                    module=name,
                    inputs=tensor_summary(args),
                    outputs=tensor_summary(output),
                )
            )
            if name == "encoder":
                actual["encoder"] = output.detach().clone()
            if name == "speaker_embedding":
                actual["speaker"] = output.detach().clone()

        return hook

    for name, module in names.items():
        hooks.append(module.register_forward_hook(capture(name)))

    def variance_input(module, args):
        actual["variance_input"] = args[0].detach().clone()

    hooks.append(model.variance_adaptor.register_forward_pre_hook(variance_input))
    try:
        output = model(
            torch.tensor([metadata["speakers"][speaker]], device="cuda"),
            ids,
            lengths,
            len(units),
        )
        wave = vocode(vocoder, output[1].transpose(1, 2))[0].cpu().numpy()
    finally:
        for hook in hooks:
            hook.remove()
    expected = actual["encoder"] + (
        actual["speaker"].unsqueeze(1) if "speaker" in actual else 0
    )
    delta = float((actual["variance_input"] - expected).abs().max())
    assert delta < 1e-5
    durations = output[5][0].cpu().tolist()
    assert sum(durations) * 256 == len(wave)
    quality = waveform(LOCAL / f"trace_{language}.wav", wave)
    write_json(
        OUT / f"trace_{language}.json",
        dict(
            test_id=test_id,
            raw_text=record["original_text"],
            normalized_text=text.normalized,
            units=units,
            token_ids=ids[0].cpu().tolist(),
            speaker_id=speaker,
            checkpoint_sha256=sha256(checkpoint),
            checkpoint_step=torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )["step"],
            model_training=model.training,
            grad_enabled=torch.is_grad_enabled(),
            events=events,
            duration_frames=durations,
            expanded_frames=sum(durations),
            speaker_addition_max_error=delta,
            pitch_normalized=output[2][0].cpu().tolist(),
            energy_normalized=output[3][0].cpu().tolist(),
            normalization_stats=metadata["stats"],
            pitch_continuous_hz=(
                output[2][0].cpu().numpy() * metadata["stats"]["pitch"][3]
                + metadata["stats"]["pitch"][2]
            ).tolist(),
            energy_stft_l2=(
                output[3][0].cpu().numpy() * metadata["stats"]["energy"][3]
                + metadata["stats"]["energy"][2]
            ).tolist(),
            vocoder_input="[1,80,T] PostNet-refined natural-log magnitude mel",
            audio=quality,
        ),
    )


def variance_figure(language, case_id, units, target, predictions):
    # Only first 30 consecutive units are displayed; all arrays are exported.
    n = min(30, len(units))
    fig, axes = plt.subplots(3, 1, figsize=(11, 8))
    for ax, key, label in zip(
        axes,
        ["duration", "pitch", "energy"],
        ["Duration (frames)", "Standardized continuous F0", "Standardized energy"],
    ):
        ax.plot(target[key][:n], label="target", color="black", linewidth=1.5)
        for role, pred in predictions.items():
            ax.plot(pred[key][:n], label=role, alpha=0.8)
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        ax.set_xticks(range(n), units[:n], rotation=90, fontsize=7)
    axes[0].legend(ncol=3)
    fig.suptitle(f"{language} {case_id}: identical aligned units, first {n} shown")
    fig.tight_layout()
    fig.savefig(OUT / f"{language}_{case_id}_variance.png", dpi=140)
    plt.close(fig)


def mel_figure(language, case_id, mels):
    # Reference-derived arrays remain local, like the legally obtained audio.
    np.savez_compressed(LOCAL / f"{language}_{case_id}_mels.npz", **mels)
    fig, axes = plt.subplots(len(mels), 1, figsize=(10, 2 * len(mels)), squeeze=False)
    for ax, (name, mel) in zip(axes[:, 0], mels.items()):
        ax.imshow(
            mel.T,
            origin="lower",
            aspect="auto",
            vmin=-10,
            vmax=2,
            extent=[0, len(mel) * 256 / 22050, 0, 80],
        )
        ax.set(title=name, ylabel="Mel band", xlabel="Time (s)")
    fig.suptitle(f"{language} {case_id}; common colour range log-mel [-10,2]")
    fig.tight_layout()
    fig.savefig(OUT / f"{language}_{case_id}_mels.png", dpi=130)
    plt.close(fig)


@torch.inference_mode()
def run():
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    before = protected_hashes()
    write_json(LOCAL / "protected_before.json", before)
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    if not torch.cuda.is_available():
        raise RuntimeError("This executed supplement requires the available GPU")
    selected = choose_rows()
    unicode_audit(selected)
    write_json(
        OUT / "protocol.json",
        dict(
            seed=2026,
            selected_ids={
                lang: [
                    dict(source_id=r["source_id"], speaker_id=r["speaker_id"])
                    for r in rows
                ]
                for lang, rows in selected.items()
            },
            selection="English: same six IDs as the completed original checkpoint probe. Telugu: first persisted validation row per speaker satisfying 5–8 seconds. No test sentences used for model selection.",
            localization="First two English rows and both Telugu rows; English checkpoint comparison uses all six English rows.",
            precision="FP32 eval/inference_mode; no optimizer, gradients or parameter updates",
            dtw="Unconstrained DTW on 4-frame mean-pooled log mels; mean path L1, not an intelligibility measure",
        ),
    )
    metrics = []
    asr_items = []
    prediction_exports = []
    for language, rows in selected.items():
        config = load_config(ROOT / language.title() / "config.yaml")
        roles = (
            {
                "step1000": ROOT / "English/checkpoints/best.pt",
                "step8000": ROOT / "English/checkpoints/selected.pt",
            }
            if language == "english"
            else {"step10500": ROOT / "Telugu/checkpoints/selected.pt"}
        )
        dataset = Features(language, "val")
        indices = {r["source_id"]: i for i, r in enumerate(dataset.rows)}
        vocoder = load_vocoder(config, "cuda")
        references = {}
        mels = {}
        variance = {}
        for index, row in enumerate(rows):
            case_id = f"{language[0]}{index+1}"
            source, rate = sf.read(
                ROOT / row["audio_path"], dtype="float32", always_2d=True
            )
            source = resample(source.mean(1), rate, 22050)
            item = dataset[indices[row["source_id"]]]
            reference = source[: len(item["mel"]) * 256]
            references[case_id] = reference
            mels[case_id] = {}
            variance[case_id] = {}
            reconstruction = (
                vocode(
                    vocoder, torch.tensor(item["mel"].T, device="cuda").unsqueeze(0)
                )[0]
                .cpu()
                .numpy()
            )
            for mode, wave in [
                ("original", reference),
                ("reference_vocoder", reconstruction),
            ]:
                path = LOCAL / f"{case_id}_{mode}.wav"
                quality = waveform(path, wave)
                metrics.append(
                    dict(
                        language=language,
                        case_id=case_id,
                        source_id=row["source_id"],
                        speaker_id=row["speaker_id"],
                        checkpoint=None,
                        mode=mode,
                        audio=quality,
                        acoustic=wave_comparison(reference, wave, True),
                    )
                )
                asr_items.append(
                    dict(
                        test_id=f"{case_id}_{mode}",
                        language=language,
                        original_text=row["raw_text"],
                        output_audio_filename=str(path.relative_to(ROOT)),
                    )
                )
            mels[case_id]["Reference mel"] = item["mel"]
            mels[case_id]["Reference-mel vocoder reanalysis"] = acoustic_features(
                reconstruction
            )[0]
        for role, checkpoint in roles.items():
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if (
                state["step"]
                != {"step1000": 1000, "step8000": 8000, "step10500": 10500}[role]
            ):
                raise ValueError(
                    f"Expected the fixed existing {role} checkpoint; do not retrain or silently replace it"
                )
            metadata = state["metadata"]
            model, _ = build_model(config, metadata)
            model.load_state_dict(state["model"])
            model.cuda().eval()
            if role == list(roles)[-1]:
                execution_trace(language, model, vocoder, metadata, checkpoint)
            for index, row in enumerate(rows):
                case_id = f"{language[0]}{index+1}"
                item = dataset[indices[row["source_id"]]]
                batch = to_device(collate([item]), "cuda")
                controlled = model(*batch[2:])
                text = frontend(row["raw_text"], language)
                units = inference_units(text)
                ids = torch.tensor(
                    [[metadata["vocab"][u] for u in units]], device="cuda"
                )
                full = model(
                    batch[2], ids, torch.tensor([len(units)], device="cuda"), len(units)
                )
                pred = dict(
                    duration=controlled[5][0].cpu().tolist(),
                    pitch=controlled[2][0].cpu().tolist(),
                    energy=controlled[3][0].cpu().tolist(),
                )
                # output[5] contains supervised durations in this controlled path;
                # use log-duration output to obtain the actual predictor estimate.
                pred["duration"] = (
                    torch.clamp(torch.round(controlled[4][0].exp() - 1), min=0)
                    .cpu()
                    .tolist()
                )
                target = {k: item[k].tolist() for k in ["duration", "pitch", "energy"]}
                variance[case_id][role] = pred
                variance_errors = dict(
                    duration_mae_frames=float(
                        np.abs(np.array(pred["duration"]) - item["duration"]).mean()
                    ),
                    log_duration_mse=float(
                        (
                            (
                                controlled[4][0].cpu().numpy()
                                - np.log1p(item["duration"])
                            )
                            ** 2
                        ).mean()
                    ),
                    pitch_standardized_mse=float(
                        ((np.array(pred["pitch"]) - item["pitch"]) ** 2).mean()
                    ),
                    energy_standardized_mse=float(
                        ((np.array(pred["energy"]) - item["energy"]) ** 2).mean()
                    ),
                )
                prediction_exports.append(
                    dict(
                        language=language,
                        case_id=case_id,
                        checkpoint=role,
                        checkpoint_sha256=sha256(checkpoint),
                        aligned_units=[
                            next(
                                u
                                for u, i in metadata["vocab"].items()
                                if i == int(token)
                            )
                            for token in item["tokens"]
                        ],
                        targets=target,
                        aligned_input_predictions=pred,
                        full_inference_units=units,
                        full_duration_frames=full[5][0].cpu().tolist(),
                        full_pitch_normalized=full[2][0].cpu().tolist(),
                        full_energy_normalized=full[3][0].cpu().tolist(),
                    )
                )
                for mode, output, aligned in [
                    ("gt_variance", controlled, True),
                    ("full_inference", full, False),
                ]:
                    mel = output[1][0].cpu().numpy()
                    wave = vocode(vocoder, output[1].transpose(1, 2))[0].cpu().numpy()
                    path = LOCAL / f"{case_id}_{role}_{mode}.wav"
                    quality = waveform(path, wave)
                    reference = item["mel"]
                    difference = dict(
                        coarse_dtw_model_mel_l1=coarse_dtw(reference, mel),
                        **(
                            {
                                "aligned_model_mel_l1": float(
                                    np.abs(reference - mel).mean()
                                )
                            }
                            if aligned
                            else {}
                        ),
                    )
                    metrics.append(
                        dict(
                            language=language,
                            case_id=case_id,
                            source_id=row["source_id"],
                            speaker_id=row["speaker_id"],
                            checkpoint=role,
                            checkpoint_sha256=sha256(checkpoint),
                            mode=mode,
                            audio=quality,
                            acoustic=wave_comparison(
                                references[case_id], wave, aligned
                            ),
                            model_mel=difference,
                            variance_errors=variance_errors,
                        )
                    )
                    asr_items.append(
                        dict(
                            test_id=f"{case_id}_{role}_{mode}",
                            language=language,
                            original_text=row["raw_text"],
                            output_audio_filename=str(path.relative_to(ROOT)),
                        )
                    )
                    if index < 2:
                        mels[case_id][f"{role}: {mode}"] = mel
                if role == list(roles)[-1] and index < 2:
                    exported = prediction_exports[-1]
                    variance_figure(
                        language,
                        case_id,
                        exported["aligned_units"],
                        target,
                        variance[case_id],
                    )
            del model, state
        for index in range(2):
            case_id = f"{language[0]}{index+1}"
            mel_figure(language, case_id, mels[case_id])
        del vocoder
    write_json(OUT / "metrics.json", metrics)
    write_json(OUT / "variance_predictions.json", prediction_exports)
    write_json(LOCAL / "asr_manifest.json", asr_items)
    after = protected_hashes()
    assert before == after, "Original assignment artifacts changed"
    write_json(
        OUT / "preservation.json",
        dict(unchanged=True, protected_files=len(before), hashes=before),
    )
    print(
        f"Executed {len(metrics)} audio conditions; original artifacts unchanged",
        flush=True,
    )


def score_asr():
    manifest = json.loads((LOCAL / "asr_manifest.json").read_text())
    transcripts = {
        r["test_id"]: r for r in json.loads((LOCAL / "asr.json").read_text())
    }
    rows = []
    for item in manifest:
        reference = words(item["original_text"], item["language"])
        hypothesis = words(transcripts[item["test_id"]]["asr_text"], item["language"])
        refchars = "".join(reference)
        hypchars = "".join(hypothesis)
        rows.append(
            dict(
                condition_id=item["test_id"],
                language=item["language"],
                reference_words=len(reference),
                word_errors=edit_distance(reference, hypothesis),
                reference_characters=len(refchars),
                character_errors=edit_distance(refchars, hypchars),
            )
        )
    write_json(
        OUT / "asr_scores.json",
        dict(
            model="Whisper large-v3, pinned revision in local asr.json",
            normalization="Existing comparison words(); character distance joins these normalized words, retaining Telugu marks and ignoring spaces",
            note="Automatic transcription proxy; controls and small samples cannot establish naturalness or causal mechanisms",
            conditions=rows,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-asr", action="store_true")
    args = parser.parse_args()
    if args.score_asr:
        score_asr()
    else:
        run()
