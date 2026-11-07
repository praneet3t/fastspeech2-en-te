"""Record the runtime without reading credentials or shell environment values."""

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from shared.io import ROOT, write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output", type=Path, default=ROOT / "artifacts/setup_environment.json"
)
args = parser.parse_args()

record = dict(
    python=sys.version,
    platform=platform.platform(),
    torch=torch.__version__,
    cuda_runtime=torch.version.cuda,
    cuda_available=torch.cuda.is_available(),
    disk_free_bytes=shutil.disk_usage(ROOT).free,
)
if torch.cuda.is_available():
    record.update(
        gpu=torch.cuda.get_device_name(),
        vram_bytes=torch.cuda.get_device_properties(0).total_memory,
    )
record["nvidia_smi"] = (
    subprocess.check_output(["nvidia-smi"], text=True)
    if shutil.which("nvidia-smi")
    else None
)
write_json(args.output, record)
print(json.dumps(record, indent=2))
