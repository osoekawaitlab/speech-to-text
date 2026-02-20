from typing import Any, Dict, List

from pytest_mock import MockerFixture
from starlette.testclient import TestClient

from ols2t.core import SpeechToTextCore
from ols2t.interfaces.http_api import HttpApi
from ols2t.models import BaseStream, Segment
from ols2t.settings import HttpApiSettings


def test_http_api_has_app(mocker: MockerFixture) -> None:
    from fastapi import FastAPI

    mock_core = mocker.MagicMock(spec=SpeechToTextCore)
    settings = HttpApiSettings()
    http_api = HttpApi(core=mock_core, settings=settings)
    assert isinstance(http_api.app, FastAPI)


def test_http_api_settings(mocker: MockerFixture) -> None:
    mock_core = mocker.MagicMock(spec=SpeechToTextCore)
    settings = HttpApiSettings()
    http_api = HttpApi(core=mock_core, settings=settings)
    assert http_api.settings.host == "0.0.0.0"
    assert http_api.settings.port == 8000


def test_http_api_run_calls_uvicorn(mocker: MockerFixture) -> None:
    mock_core = mocker.MagicMock(spec=SpeechToTextCore)
    settings = HttpApiSettings()
    http_api = HttpApi(core=mock_core, settings=settings)
    mock_uvicorn_run = mocker.patch("ols2t.interfaces.http_api.uvicorn.run")
    http_api.run()
    mock_uvicorn_run.assert_called_once_with(http_api.app, host="0.0.0.0", port=8000)


def test_post_transcribe(mocker: MockerFixture) -> None:
    mock_core = mocker.MagicMock(spec=SpeechToTextCore)
    mock_core.transcribe.return_value = iter([Segment(text="こんにちは", start=0.0, end=2.0, probability=0.9)])
    settings = HttpApiSettings()
    http_api = HttpApi(core=mock_core, settings=settings)
    client = TestClient(http_api.app)
    response = client.post(
        "/transcribe",
        files={"file": ("hello.wav", b"fake audio data", "audio/wav")},
    )
    assert response.status_code == 200
    data: List[Dict[str, Any]] = response.json()
    assert len(data) == 1
    assert data[0]["text"] == "こんにちは"
    assert data[0]["start"] == 0.0
    assert data[0]["end"] == 2.0
    assert data[0]["probability"] == 0.9


def test_post_transcribe_multiple_segments(mocker: MockerFixture) -> None:
    mock_core = mocker.MagicMock(spec=SpeechToTextCore)
    mock_core.transcribe.return_value = iter(
        [
            Segment(text="こんにちは", start=0.0, end=2.0, probability=0.9),
            Segment(text="世界", start=2.0, end=3.0, probability=0.8),
        ]
    )
    settings = HttpApiSettings()
    http_api = HttpApi(core=mock_core, settings=settings)
    client = TestClient(http_api.app)
    response = client.post(
        "/transcribe",
        files={"file": ("test.wav", b"fake audio data", "audio/wav")},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["text"] == "こんにちは"
    assert data[1]["text"] == "世界"


def test_ws_transcribe_receives_segments(mocker: MockerFixture) -> None:
    mock_core = mocker.MagicMock(spec=SpeechToTextCore)

    def fake_transcribe(input_stream: BaseStream) -> Any:
        return iter(
            [
                Segment(text="こんにちは", start=0.0, end=2.0, probability=0.9),
                Segment(text="世界", start=2.0, end=3.0, probability=0.8),
            ]
        )

    mock_core.transcribe.side_effect = fake_transcribe
    settings = HttpApiSettings()
    http_api = HttpApi(core=mock_core, settings=settings)
    client = TestClient(http_api.app)
    received: List[Dict[str, Any]] = []
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_bytes(b"fake audio chunk")
        ws.send_bytes(b"")  # signal end of audio
        while True:
            data = ws.receive_json()
            if "done" in data:
                break
            received.append(data)
    assert len(received) == 2
    assert received[0]["text"] == "こんにちは"
    assert received[1]["text"] == "世界"
    mock_core.transcribe.assert_called_once()
