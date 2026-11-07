"""Release packaging risks: offline resources, path safety and evidence preservation."""

from html.parser import HTMLParser
import json
import subprocess
import sys

from shared.io import ROOT


class Resources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.audio = []
        self.images = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "audio":
            self.audio.append(attrs)
        if tag == "img":
            self.images.append(attrs)
        if tag == "a":
            self.links.append(attrs)


def test_viewer_resources():
    rows = json.loads((ROOT / "data/test_sentences.json").read_text())
    for name in ["results.html"]:
        parser = Resources()
        parser.feed((ROOT / name).read_text())
        assert len(parser.audio) == 10 and len(parser.images) == 34
        assert {r["data-test"] for r in parser.audio} == {r["test_id"] for r in rows}
        for audio in parser.audio:
            source = audio["src"]
            assert (ROOT / source).is_file()
        for image in parser.images:
            source = image["src"]
            assert (ROOT / source).is_file()


def test_public_artifact_manifest_is_complete():
    data = json.loads((ROOT / "analysis/release/artifact_manifest.json").read_text())
    ids = {a["id"] for a in data["artifacts"]}
    assert {
        "english_selected",
        "telugu_selected",
        "vocoder",
    } <= ids
    for a in data["artifacts"]:
        assert not a["requires_authentication"] and a["publicly_available"]
        assert len(a["sha256"]) == 64 and a["bytes"] > 0
        expected = (
            "https://raw.githubusercontent.com/ming024/"
            if a["id"] == "vocoder"
            else "https://github.com/praneet3t/fastspeech2-en-te/releases/"
        )
        assert a["download_url"].startswith(expected)


def test_downloader_rejects_traversal_before_network_access(tmp_path):
    manifest = tmp_path / "bad.json"
    manifest.write_text(
        json.dumps(dict(artifacts=[dict(id="unsafe", path="../../escape.pt")]))
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/download_artifacts.py"),
            "--manifest",
            str(manifest),
            "--directory",
            str(tmp_path / "restore"),
            "--all",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "Unsafe artifact destination" in result.stderr
    assert not (tmp_path.parent / "escape.pt").exists()


def test_environment_probe_does_not_replace_experiment_record(tmp_path):
    path = ROOT / "analysis/environment.json"
    before = path.read_bytes()
    output = tmp_path / "environment.json"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/environment.py"), "--output", str(output)],
        check=True,
        capture_output=True,
    )
    assert path.read_bytes() == before
    assert json.loads(output.read_text())["torch"]
