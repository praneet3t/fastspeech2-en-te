"""Download pinned official dataset files, respecting Hugging Face access gates."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError
from shared.io import ROOT, write_json

SOURCES = {
    "english": ("ai4bharat/Svarah", "ebbf7777fe771490696a3f7b007097606fa8c924"),
    "telugu": ("SPRINGLab/IndicTTS_Telugu", "7ab3c18491bef3f5d4e820c2701b5f78f54031d6"),
}


def download(language):
    repo, revision = SOURCES[language]
    try:
        path = snapshot_download(
            repo,
            repo_type="dataset",
            revision=revision,
            local_dir=ROOT / "data/raw" / language,
            max_workers=4,
            allow_patterns=["*.parquet", "README.md"],
        )
    except GatedRepoError as error:
        raise SystemExit(
            "Svarah access denied. Accept its conditions at https://huggingface.co/datasets/ai4bharat/Svarah "
            "and run `hf auth login` locally with an authorized token. Do not commit credentials."
        ) from error
    write_json(
        ROOT / "analysis" / f"{language}_source.json",
        dict(
            dataset=repo,
            revision=revision,
            local_path=str(Path(path).relative_to(ROOT)),
        ),
    )
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("language", choices=SOURCES)
    download(parser.parse_args().language)
