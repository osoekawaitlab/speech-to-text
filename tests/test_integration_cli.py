import json
import os
import subprocess
import tempfile

import pytest


@pytest.mark.slow
def test_cli_run_transcribe(
    fixture_dir: str, patch_whisper_small_environment_variable: None, patch_cli_environment_variable: None
) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        output_file_name = "output.jsonl"
        error_code = subprocess.run(
            [
                "ols2t",
                "transcribe",
                os.path.join(fixture_dir, "hello_ja.wav"),
                os.path.join(tempdir, output_file_name),
            ],
            env=os.environ,
        )
        assert error_code.returncode == 0
        with open(os.path.join(tempdir, output_file_name), "r") as output_file:
            data = [json.loads(line) for line in output_file]
        assert len(data) == 1
        assert data[0]["text"] == "こんにちは"


def test_cli_has_version() -> None:
    from ols2t import __version__

    version = subprocess.run(["ols2t", "--version"], stdout=subprocess.PIPE, check=True)
    assert version.stdout.decode().strip() == __version__


@pytest.mark.slow
def test_cli_transcribe_with_settings_option_json(fixture_dir: str) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        output_file_name = "output.jsonl"
        error_code = subprocess.run(
            [
                "ols2t",
                "--settings",
                os.path.join(fixture_dir, "tiny_ja_settings.json"),
                "transcribe",
                os.path.join(fixture_dir, "hello_ja.wav"),
                os.path.join(tempdir, output_file_name),
            ],
            env=os.environ,
        )
        assert error_code.returncode == 0
        with open(os.path.join(tempdir, output_file_name), "r") as output_file:
            data = [json.loads(line) for line in output_file]
        assert len(data) == 5
        assert "".join(s["text"] for s in data) == "ここに位置は"


@pytest.mark.slow
def test_cli_transcribe_with_settings_option_yaml(fixture_dir: str) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        output_file_name = "output.jsonl"
        error_code = subprocess.run(
            [
                "ols2t",
                "--settings",
                os.path.join(fixture_dir, "tiny_ja_settings.yaml"),
                "transcribe",
                os.path.join(fixture_dir, "hello_ja.wav"),
                os.path.join(tempdir, output_file_name),
            ],
            env=os.environ,
        )
        assert error_code.returncode == 0
        with open(os.path.join(tempdir, output_file_name), "r") as output_file:
            data = [json.loads(line) for line in output_file]
        assert len(data) == 5
        assert "".join(s["text"] for s in data) == "ここに位置は"
