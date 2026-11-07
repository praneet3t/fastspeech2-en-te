"""Independent FastSpeech 2 training with atomic, resumable checkpoints."""

import argparse
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from shared.io import ROOT, load_config, write_json, sha256
from shared.batches import Features, collate, to_device
from shared.model import build_model, load_metadata

LOSS_NAMES = ["total", "mel", "postnet_mel", "pitch", "energy", "duration"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    step,
    epoch,
    offset,
    best,
    config,
    elapsed,
    generator,
):
    checkpoint = dict(
        metadata=load_metadata(config),
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        scaler=scaler.state_dict(),
        step=step,
        epoch=epoch,
        batch_offset=offset,
        best_validation=best,
        config=config,
        elapsed_seconds=elapsed,
        python_rng=random.getstate(),
        numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all(),
        loader_rng=generator.get_state(),
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def restore(path, model, optimizer=None, scaler=None):
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Run real training first."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    return checkpoint


def learning_rate(step, config):
    warmup = config["training"]["warmup_steps"]
    hidden = config["model"]["transformer"]["encoder_hidden"]
    return (
        config["training"]["learning_rate_scale"]
        * hidden**-0.5
        * min(step**-0.5, step * warmup**-1.5)
    )


def append_metric(path, value):
    with open(path, "a") as f:
        f.write(json.dumps(value) + "\n")
        f.flush()


@torch.inference_mode()
def validate(model, criterion, loader, device):
    model.eval()
    # Each component is accumulated with its actual mask denominator. Total is
    # the sum of these corpus means, matching the five-term training objective.
    sums = np.zeros(5)
    denominators = np.zeros(5)
    duration_abs = 0.0
    unit_count = 0
    for cpu_batch in loader:
        batch = to_device(cpu_batch, device)
        output = model(*batch[2:])
        losses = criterion(batch, output)
        if not all(torch.isfinite(x) for x in losses):
            raise FloatingPointError("Non-finite validation loss")
        units = int(batch[4].sum())
        frames = int(batch[7].sum())
        weights = np.array([frames * 80, frames * 80, units, units, units])
        sums += np.array([x.item() for x in losses[1:]]) * weights
        denominators += weights
        mask = ~output[6]
        duration_prediction = torch.clamp(torch.round(torch.exp(output[4]) - 1), min=0)
        duration_abs += (
            (duration_prediction - batch[11]).abs().masked_select(mask).sum().item()
        )
        unit_count += units
    components = sums / denominators
    result = dict(zip(LOSS_NAMES, [float(components.sum())] + components.tolist()))
    result["duration_mae_frames"] = duration_abs / unit_count
    result["validation_mode"] = (
        "teacher-forced ground-truth duration, pitch and energy conditioning"
    )
    model.train()
    return result


def epoch_order(lengths, seed, epoch, batch_size, bucket_multiplier=1):
    """Seeded shuffle, then sort only bounded groups to limit padding.

    Multiplier four matches upstream's group-of-four-batches strategy. One
    preserves the unbucketed English experiment already executed.
    """
    order = torch.randperm(
        len(lengths), generator=torch.Generator().manual_seed(seed + epoch)
    ).tolist()
    group_size = batch_size * bucket_multiplier
    if bucket_multiplier > 1:
        order = [
            index
            for start in range(0, len(order), group_size)
            for index in sorted(
                order[start : start + group_size],
                key=lambda i: lengths[i],
                reverse=True,
            )
        ]
    return order


def main(default_language):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=f"{default_language}/config.yaml")
    parser.add_argument("--resume", nargs="?", const="latest")
    parser.add_argument(
        "--steps",
        type=int,
        help="Override total optimizer steps; recorded in checkpoint config",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if config["language"] != default_language.lower():
        raise ValueError("Entry point/config language mismatch")
    if args.steps:
        config["training"]["total_steps"] = args.steps
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU required for this full experiment; no CPU smoke run will be reported as training"
        )
    torch.set_num_threads(4)
    seed_all(config["seed"])
    device = torch.device("cuda")
    options = config["training"]
    directory = ROOT / options["checkpoint_dir"]
    directory.mkdir(parents=True, exist_ok=True)
    logdir = ROOT / "analysis" / config["language"]
    logdir.mkdir(parents=True, exist_ok=True)
    train_data = Features(config["language"], "train")
    val_data = Features(config["language"], "val")
    generator = torch.Generator().manual_seed(config["seed"])
    # Dedicated worker RNG prevents loader creation from changing model dropout RNG.
    worker_generator = torch.Generator().manual_seed(config["seed"] + 1)
    model, criterion = build_model(config)
    model.to(device)
    criterion.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0, betas=options["betas"], eps=options["epsilon"]
    )
    use_fp16 = options["precision"] == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        options["precision"]
    ]
    step, epoch, offset, best, elapsed = 0, 0, 0, float("inf"), 0.0
    if args.resume:
        path = directory / "latest.pt" if args.resume == "latest" else Path(args.resume)
        checkpoint = restore(path, model, optimizer, scaler)
        prior = checkpoint["config"]
        if (
            checkpoint.get("metadata", {}).get(
                "feature_hashes", load_metadata(config)["feature_hashes"]
            )
            != load_metadata(config)["feature_hashes"]
        ):
            raise ValueError("Preprocessed features differ from resumed checkpoint")
        if prior["dataset"] != config["dataset"] or prior["model"] != config["model"]:
            raise ValueError("Resume dataset/model configuration differs")
        step, epoch, offset, best, elapsed = [
            checkpoint[k]
            for k in [
                "step",
                "epoch",
                "batch_offset",
                "best_validation",
                "elapsed_seconds",
            ]
        ]
        generator.set_state(checkpoint["loader_rng"])
        metrics_path = logdir / "metrics.jsonl"
        if metrics_path.exists():
            valid = [
                json.loads(line)
                for line in metrics_path.read_text().splitlines()
                if line.strip()
            ]
            valid = [row for row in valid if row["step"] <= step]
            temporary = metrics_path.with_suffix(".tmp")
            temporary.write_text("".join(json.dumps(row) + "\n" for row in valid))
            os.replace(temporary, metrics_path)
    elif (logdir / "metrics.jsonl").exists():
        raise FileExistsError(
            "Metrics already exist. Resume the checkpoint or explicitly archive the previous run."
        )
    val_loader = DataLoader(
        val_data,
        batch_size=options["batch_size"],
        collate_fn=collate,
        shuffle=False,
        num_workers=options["workers"],
        pin_memory=True,
        generator=worker_generator,
    )
    write_json(
        logdir / "run.json",
        dict(
            config=config,
            parameters=sum(p.numel() for p in model.parameters()),
            gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            source_hashes={
                name: sha256(ROOT / "data/features" / config["language"] / name)
                for name in [
                    "stats.json",
                    "vocab.json",
                    "speakers.json",
                    "train.jsonl",
                    "val.jsonl",
                ]
            },
        ),
    )
    started = time.monotonic()
    model.train()
    print(
        f'{config["language"]}: {sum(p.numel() for p in model.parameters()):,} parameters, resume step={step}',
        flush=True,
    )
    accumulated = 0
    meter = np.zeros(6)
    meter_count = 0
    stop_requested = False
    validation_history = []
    if (logdir / "metrics.jsonl").exists():
        validation_history = [
            json.loads(line)
            for line in (logdir / "metrics.jsonl").read_text().splitlines()
            if json.loads(line).get("split") == "val"
        ]
    optimizer.zero_grad(set_to_none=True)
    while step < options["total_steps"]:
        # Epoch permutation can be rebuilt exactly from seed+epoch and offset.
        order = epoch_order(
            [row["frames"] for row in train_data.rows],
            config["seed"],
            epoch,
            options["batch_size"],
            options.get("length_bucket_multiplier", 1),
        )
        # Subset avoids consuming worker RNG/dropout on already-processed batches.
        order = order[offset * options["batch_size"] :]
        loader = DataLoader(
            torch.utils.data.Subset(train_data, order),
            batch_size=options["batch_size"],
            shuffle=False,
            collate_fn=collate,
            num_workers=options["workers"],
            pin_memory=True,
            generator=worker_generator,
        )
        for cpu_batch in loader:
            batch = to_device(cpu_batch, device)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                output = model(*batch[2:])
                losses = criterion(batch, output)
                loss = losses[0] / options["gradient_accumulation"]
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite train loss after step {step}; latest checkpoint retained"
                )
            scaler.scale(loss).backward()
            accumulated += 1
            offset += 1
            meter += np.array([x.item() for x in losses])
            meter_count += 1
            if accumulated < options["gradient_accumulation"]:
                continue
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), options["gradient_clip"], error_if_nonfinite=True
            )
            lr = learning_rate(step + 1, config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            step += 1
            elapsed_now = elapsed + time.monotonic() - started
            if step % options["log_interval"] == 0:
                record = dict(
                    split="train",
                    step=step,
                    epoch=epoch,
                    elapsed_seconds=elapsed_now,
                    lr=lr,
                    gradient_norm=float(gradient_norm),
                    **dict(zip(LOSS_NAMES, (meter / meter_count).tolist())),
                )
                append_metric(logdir / "metrics.jsonl", record)
                print(json.dumps(record), flush=True)
                meter[:] = 0
                meter_count = 0
            is_best = False
            if (
                step % options["validation_interval"] == 0
                or step == options["total_steps"]
            ):
                result = validate(model, criterion, val_loader, device)
                record = dict(
                    split="val",
                    step=step,
                    epoch=epoch,
                    elapsed_seconds=elapsed + time.monotonic() - started,
                    **result,
                )
                append_metric(logdir / "metrics.jsonl", record)
                print(json.dumps(record), flush=True)
                if result["total"] < best:
                    best = result["total"]
                    is_best = True
                validation_history.append(record)
                patience = options.get("early_stopping_patience", 8)
                if (
                    step >= options.get("min_steps", 8000)
                    and len(validation_history) > patience
                ):
                    old_best = min(
                        row["total"] for row in validation_history[:-patience]
                    )
                    recent_best = min(
                        row["total"] for row in validation_history[-patience:]
                    )
                    improvement = (old_best - recent_best) / old_best
                    if improvement < options.get("relative_min_improvement", 0.001):
                        stop_requested = True
                        print(
                            f"Validation plateau: relative improvement {improvement:.6f} over {patience} evaluations",
                            flush=True,
                        )
            if (
                step % options["checkpoint_interval"] == 0
                or step == options["total_steps"]
                or stop_requested
            ):
                save_checkpoint(
                    directory / "latest.pt",
                    model,
                    optimizer,
                    scaler,
                    step,
                    epoch,
                    offset,
                    best,
                    config,
                    elapsed + time.monotonic() - started,
                    generator,
                )
            if is_best:
                save_checkpoint(
                    directory / "best.pt",
                    model,
                    optimizer,
                    scaler,
                    step,
                    epoch,
                    offset,
                    best,
                    config,
                    elapsed + time.monotonic() - started,
                    generator,
                )
            if step >= options["total_steps"] or stop_requested:
                break
        epoch += 1
        offset = 0
        if stop_requested:
            break
    write_json(
        logdir / "training_complete.json",
        dict(
            step=step,
            stop_reason=(
                "validation_plateau" if stop_requested else "configured_step_limit"
            ),
            best_validation=best,
            elapsed_seconds=elapsed + time.monotonic() - started,
            checkpoint=str((directory / "best.pt").relative_to(ROOT)),
        ),
    )
