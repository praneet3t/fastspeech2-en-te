"""Download public acoustic checkpoints and the original upstream vocoder."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from urllib.request import Request, urlopen
from zipfile import ZipFile


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "analysis/release/artifact_manifest.json",
    )
    parser.add_argument("--directory", type=Path, default=root)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Retrieve all entries in the manifest (the default)",
    )
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    destination = args.directory.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for item in manifest["artifacts"]:
        relative = PurePosixPath(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe artifact destination")
        target = destination / str(relative)
        if not target.resolve().is_relative_to(destination):
            raise ValueError("Destination escapes through a symlink")
        if target.exists() and not args.force_download:
            if (
                target.stat().st_size == item["bytes"]
                and digest(target) == item["sha256"]
            ):
                print("Already verified:", relative, flush=True)
                continue
            raise ValueError(
                f"Existing artifact differs: {target}; preserve it before using --force-download"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="fastspeech2-download-", dir=target.parent
        ) as temporary:
            received = Path(temporary) / "download"
            request = Request(
                item["download_url"], headers={"User-Agent": "fastspeech2-en-te"}
            )
            with (
                urlopen(request, timeout=120) as response,
                received.open("wb") as handle,
            ):
                shutil.copyfileobj(response, handle)
            if "archive_member" in item:
                if digest(received) != item["archive_sha256"]:
                    raise ValueError("Vocoder archive integrity failure")
                unpacked = Path(temporary) / "vocoder"
                with ZipFile(received) as archive:
                    if archive.namelist() != [item["archive_member"]]:
                        raise ValueError("Unexpected vocoder archive layout")
                    with (
                        archive.open(item["archive_member"]) as source,
                        unpacked.open("wb") as output,
                    ):
                        shutil.copyfileobj(source, output)
                received = unpacked
            if (
                received.stat().st_size != item["bytes"]
                or digest(received) != item["sha256"]
            ):
                raise ValueError(f"Download integrity failure: {item['id']}")
            os.replace(received, target)
        print("Downloaded and SHA256 verified:", relative, flush=True)


if __name__ == "__main__":
    main()
