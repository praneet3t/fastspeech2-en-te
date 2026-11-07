#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -r references/python-lock.txt --extra-index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m nltk.downloader averaged_perceptron_tagger averaged_perceptron_tagger_eng cmudict
.venv/bin/python scripts/environment.py --output artifacts/setup_environment.json
