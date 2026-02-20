import glob
import os
from typing import Any, Dict, List

import pytest
from starlette.testclient import TestClient

import ols2t
from ols2t.interfaces.http_api import HttpApi
from ols2t.settings import HttpApiSettings


@pytest.fixture
def tiny_ja_core() -> ols2t.SpeechToTextCore:
    settings = ols2t.SpeechToTextCoreSettings(
        speech_to_text_model_settings={"type": "WHISPER", "path_or_model_size": "tiny", "language": "ja"}
    )
    return ols2t.SpeechToTextCore.create(settings=settings)


@pytest.fixture
def http_api_client(tiny_ja_core: ols2t.SpeechToTextCore) -> TestClient:
    settings = HttpApiSettings()
    http_api = HttpApi(core=tiny_ja_core, settings=settings)
    return TestClient(http_api.app)


@pytest.mark.slow
def test_post_transcribe_hello_ja(http_api_client: TestClient, fixture_dir: str) -> None:
    hello_wav_path = os.path.join(fixture_dir, "hello_ja.wav")
    with open(hello_wav_path, "rb") as f:
        response = http_api_client.post(
            "/transcribe",
            files={"file": ("hello_ja.wav", f, "audio/wav")},
        )
    assert response.status_code == 200
    data: List[Dict[str, Any]] = response.json()
    assert len(data) > 0
    text = "".join(s["text"] for s in data)
    assert "こんにちは" in text or "ここに位置は" in text


@pytest.mark.slow
def test_ws_transcribe_webm_chunks(http_api_client: TestClient, fixture_dir: str) -> None:
    chunk_dir = os.path.join(fixture_dir, "webm_chunks")
    chunk_files = sorted(glob.glob(os.path.join(chunk_dir, "webm_chunk_*.bin")))
    assert len(chunk_files) > 0

    received: List[Dict[str, Any]] = []
    with http_api_client.websocket_connect("/ws/transcribe") as ws:
        for chunk_file in chunk_files:
            with open(chunk_file, "rb") as f:
                ws.send_bytes(f.read())
        ws.send_bytes(b"")  # signal end of audio
        while True:
            data = ws.receive_json()
            if "done" in data:
                break
            received.append(data)
    assert len(received) > 0
    text = "".join(s["text"] for s in received)
    assert len(text) > 0
