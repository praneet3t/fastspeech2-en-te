import json
import copy
import numpy as np
import pytest
import torch
from shared.io import ROOT, load_config
from shared.model import FastSpeech2, FastSpeech2Loss
from shared.batches import collate
from shared.features import acoustic_features
from shared.train import restore, learning_rate


@pytest.fixture
def model_and_config(tmp_path):
    (tmp_path / "speakers.json").write_text(json.dumps({"female": 0, "male": 1}))
    (tmp_path / "stats.json").write_text(
        json.dumps({"pitch": [-2, 2, 200, 50], "energy": [-2, 2, 5, 2]})
    )
    preprocess = dict(
        path=dict(preprocessed_path=str(tmp_path)),
        preprocessing=dict(
            mel=dict(n_mel_channels=80),
            pitch=dict(feature="phoneme_level"),
            energy=dict(feature="phoneme_level"),
        ),
    )
    config = copy.deepcopy(load_config(ROOT / "Telugu/config.yaml")["model"])
    config["vocab_size"] = 20
    config["max_seq_len"] = 8
    config["transformer"].update(
        encoder_layer=1,
        decoder_layer=1,
        encoder_hidden=32,
        decoder_hidden=32,
        conv_filter_size=64,
    )
    config["variance_predictor"]["filter_size"] = 32
    return FastSpeech2(preprocess, config), FastSpeech2Loss(preprocess, config)


def small_batch():
    items = []
    for n in [4, 3]:
        duration = np.arange(1, n + 1)
        items.append(
            dict(
                id=str(n),
                speaker=n % 2,
                tokens=np.arange(1, n + 1),
                duration=duration,
                mel=np.zeros((duration.sum(), 80), dtype="float32"),
                pitch=np.zeros(n),
                energy=np.zeros(n),
            )
        )
    return collate(items)


def test_shapes_masks_finite_loss_backward(model_and_config):
    model, criterion = model_and_config
    batch = small_batch()
    output = model(*batch[2:])
    assert output[0].shape == (
        2,
        10,
        80,
    )  # Exceeds configured positional table: no truncation.
    assert output[2].shape == output[3].shape == output[4].shape == (2, 4)
    assert model.speaker_emb.weight.shape == (2, 32)
    assert output[6][1, 3] and output[7][1, 6:].all()
    losses = criterion(batch, output)
    assert all(torch.isfinite(x) for x in losses)
    losses[0].backward()
    assert model.encoder.src_word_emb.weight.grad is not None


def test_length_regulator(model_and_config):
    model, _ = model_and_config
    lr = model.variance_adaptor.length_regulator
    source = torch.arange(12).reshape(1, 3, 4).float()
    output, lengths = lr(source, torch.tensor([[2, 0, 1]]), None)
    assert lengths.tolist() == [3] and output.shape == (1, 3, 4)
    assert torch.equal(output[0, 0], output[0, 1]) and torch.equal(
        output[0, 2], source[0, 2]
    )
    with pytest.raises(ValueError, match="Negative"):
        lr(source, torch.tensor([[-1, 1, 2]]), None)


def test_save_load_equivalence(model_and_config, tmp_path):
    model, _ = model_and_config
    model.eval()
    batch = small_batch()
    expected = model(*batch[2:])[1].detach()
    path = tmp_path / "weights.pt"
    torch.save(dict(model=model.state_dict()), path)
    with torch.no_grad():
        model.mel_linear.weight.zero_()
    restore(path, model)
    torch.testing.assert_close(model(*batch[2:])[1], expected)
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        restore(tmp_path / "missing.pt", model)


def test_stft_frame_count_and_finite():
    samples = np.sin(2 * np.pi * 220 * np.arange(22050) / 22050).astype("float32") * 0.1
    mel, energy = acoustic_features(samples)
    assert mel.shape == (len(samples) // 256, 80) and len(energy) == len(mel)
    assert np.isfinite(mel).all() and (energy > 0).all()


def test_lr_resume_is_step_determined():
    config = load_config(ROOT / "Telugu/config.yaml")
    assert learning_rate(4000, config) > learning_rate(2000, config)
    assert learning_rate(8000, config) < learning_rate(4000, config)


def test_full_optimizer_rng_resume(tmp_path, monkeypatch):
    import shared.train as train

    monkeypatch.setattr(train, "load_metadata", lambda config: {"feature_hashes": {}})
    train.seed_all(2026)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    def update():
        optimizer.zero_grad()
        x = torch.randn(4, 3)
        loss = model(x).square().mean()
        loss.backward()
        optimizer.step()
        return loss.detach().clone()

    update()
    generator = torch.Generator().manual_seed(2026)
    path = tmp_path / "resume.pt"
    train.save_checkpoint(
        path, model, optimizer, scaler, 1, 0, 2, 1.0, {}, 2.0, generator
    )
    expected_loss = update()
    expected = {k: v.detach().clone() for k, v in model.state_dict().items()}
    checkpoint = train.restore(path, model, optimizer, scaler)
    actual_loss = update()
    assert checkpoint["step"] == 1 and checkpoint["batch_offset"] == 2
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    for key in expected:
        torch.testing.assert_close(
            model.state_dict()[key], expected[key], rtol=0, atol=0
        )


def test_length_grouping_is_reproducible_permutation():
    from shared.train import epoch_order

    lengths = list(range(101))
    order = epoch_order(lengths, 2026, 2, 8, 4)
    assert sorted(order) == list(range(101))
    assert order == epoch_order(lengths, 2026, 2, 8, 4)
    assert order != epoch_order(lengths, 2026, 3, 8, 4)
    for start in range(0, len(order), 32):
        assert order[start : start + 32] == sorted(
            order[start : start + 32], reverse=True
        )
