"""Separate resumable stages: audit/select -> real MFA alignment -> features."""

import argparse
from shared.io import load_config


def main(default_language):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=f"{default_language}/config.yaml")
    parser.add_argument(
        "--stage", choices=["data", "corpus", "align", "features", "all"], default="all"
    )
    parser.add_argument("--redo-audit", action="store_true")
    parser.add_argument("--train-aligner", action="store_true")
    parser.add_argument(
        "--clean-aligner",
        action="store_true",
        help="Discard MFA work state after changing corpus/dictionary/config",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if config["language"] != default_language.lower():
        raise ValueError("Entry point and config language differ")
    if args.stage in ("data", "all"):
        from shared.data import prepare_data

        prepare_data(config, args.redo_audit)
    if args.stage == "corpus":
        from shared.alignment import prepare_corpus

        prepare_corpus(config)
    if args.stage in ("align", "all"):
        from shared.alignment import run_mfa

        run_mfa(config, args.train_aligner, args.clean_aligner)
    if args.stage in ("features", "all"):
        from shared.features import prepare_features

        prepare_features(config)
