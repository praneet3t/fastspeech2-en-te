"""Retrieve the universal checkpoint distributed by pinned ming024/FastSpeech2.

The checkpoint is kept outside Git. Record its hash and original distribution.
The repository's MIT licenses cover source; checkpoint-specific training-data
rights are not inferred from a source-code license.
"""

import sys
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.io import ROOT, sha256, write_json

commit = "d4e79eb52e8b01d24703b2dfc0385544092958f3"
folder = ROOT / "artifacts/vocoder"
folder.mkdir(parents=True, exist_ok=True)
url = f"https://raw.githubusercontent.com/ming024/FastSpeech2/{commit}/hifigan/generator_universal.pth.tar.zip"
archive = folder / "generator_universal.pth.tar.zip"
if not archive.exists():
    urlretrieve(url, archive)
with ZipFile(archive) as z:
    if z.namelist() != ["generator_universal.pth.tar"]:
        raise ValueError("Unexpected vocoder archive layout")
    z.extractall(folder)
write_json(
    ROOT / "analysis/vocoder_provenance.json",
    dict(
        url=url,
        upstream_commit=commit,
        sha256=sha256(folder / "generator_universal.pth.tar"),
        archive_sha256=sha256(archive),
        architecture="HiFi-GAN V1 universal checkpoint distributed by ming024",
        conditioning="22050 Hz, FFT/window 1024, hop 256, 80 Slaney mel bands 0–8000 Hz, natural log magnitude clamp 1e-5",
        distribution="Original upstream download; this project does not redistribute pretrained weights",
    ),
)
