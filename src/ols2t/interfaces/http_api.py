import asyncio
import os
import queue as stdlib_queue
import tempfile
from multiprocessing import Event as MPEvent
from multiprocessing import Queue as MPQueue
from typing import Any, Dict, List

try:
    import uvicorn
    from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
except ImportError:
    raise ImportError(
        "fastapi and uvicorn are required for the HTTP API interface. " "Install them with: pip install ols2t[http]"
    )

from ..core import SpeechToTextCore
from ..models import BytesChunkStream, FileStream, Segment
from ..settings import HttpApiSettings
from .base import BaseInterface


class HttpApi(BaseInterface):
    def __init__(self, core: SpeechToTextCore, settings: HttpApiSettings) -> None:
        super().__init__(core=core)
        self._settings = settings
        self._app = self._create_app()

    @property
    def settings(self) -> HttpApiSettings:
        return self._settings

    @property
    def app(self) -> FastAPI:
        return self._app

    def _create_app(self) -> FastAPI:
        app = FastAPI()
        core = self.core

        @app.post("/transcribe")
        async def transcribe(file: UploadFile) -> List[Dict[str, Any]]:
            content = await file.read()
            suffix = ""
            if file.filename:
                _, suffix = os.path.splitext(file.filename)
            if not suffix:
                suffix = ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            try:
                stream = FileStream(path=tmp_path)
                loop = asyncio.get_event_loop()
                segments: List[Segment] = await loop.run_in_executor(
                    None, lambda: list(core.transcribe(input_stream=stream))
                )
                return [s.model_dump() for s in segments]
            finally:
                os.unlink(tmp_path)

        @app.websocket("/ws/transcribe")
        async def ws_transcribe(websocket: WebSocket) -> None:
            await websocket.accept()
            chunk_queue: "MPQueue[bytes]" = MPQueue(maxsize=256)
            stop_event = MPEvent()
            stream = BytesChunkStream(chunk_queue=chunk_queue, stop_event=stop_event)

            seg_q: stdlib_queue.Queue[Segment | None] = stdlib_queue.Queue()

            def _transcribe() -> None:
                try:
                    for segment in core.transcribe(input_stream=stream):
                        seg_q.put(segment)
                finally:
                    seg_q.put(None)

            loop = asyncio.get_event_loop()
            transcribe_future = loop.run_in_executor(None, _transcribe)

            async def receive_audio() -> None:
                try:
                    while True:
                        data = await websocket.receive_bytes()
                        if len(data) == 0:
                            break
                        chunk_queue.put(data, block=True, timeout=5.0)
                except WebSocketDisconnect:
                    pass
                finally:
                    stop_event.set()

            async def send_segments() -> None:
                while True:
                    try:
                        segment = await loop.run_in_executor(None, seg_q.get, True, 1.0)
                    except stdlib_queue.Empty:
                        continue
                    if segment is None:
                        break
                    await websocket.send_json(segment.model_dump())
                await websocket.send_json({"done": True})

            try:
                await asyncio.gather(receive_audio(), send_segments())
            except Exception:
                stop_event.set()
            finally:
                await transcribe_future

        return app

    def run(self) -> None:
        uvicorn.run(self.app, host=self._settings.host, port=self._settings.port)
