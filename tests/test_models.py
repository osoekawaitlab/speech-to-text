import glob
import time
from collections.abc import Iterable
from multiprocessing import Event as MPEvent
from multiprocessing import Process
from multiprocessing import Queue as MPQueue
from multiprocessing.synchronize import Event as EventClass

import numpy as np
from numpy.typing import NDArray
from pytest_mock import MockerFixture

from ols2t.models import BytesChunkStream, MicrophoneStream


def test_microphone_stream(
    mocker: MockerFixture,
    longtext_all_decoded_one_second_chunks: Iterable[NDArray[np.float32]],
    longtext_all_decoded_fixture_path: str,
) -> None:
    PyAudio = mocker.patch("ols2t.models.PyAudio")
    PyAudio.return_value.open.return_value.read.side_effect = longtext_all_decoded_one_second_chunks
    PyAudio.return_value.open.return_value.get_read_available.return_value = 16000
    sut = MicrophoneStream()
    with open(longtext_all_decoded_fixture_path, "rb") as f:
        x = np.load(f)
    cnt = 0
    with sut as stream:
        for i, chunk in enumerate(stream):
            assert np.allclose(chunk, x[i * 16000 : (i + 1) * 16000])
            cnt += 1
            if i == 3:
                stream.stop()
    assert cnt == 4


def test_bytes_chunk_stream(
    mocker: MockerFixture,
) -> None:

    stop_event = MPEvent()

    def adding_chunks(queue: "MPQueue[bytes]", stop_event: EventClass) -> None:
        for chunk_path in sorted(glob.glob("tests/fixtures/webm_chunks/webm_chunk_*.bin")):
            with open(chunk_path, "rb") as f:
                queue.put(f.read())
                time.sleep(0.7)
        stop_event.set()

    queue: "MPQueue[bytes]" = MPQueue(maxsize=256)
    sut = BytesChunkStream(chunk_queue=queue, stop_event=stop_event)
    with open("tests/fixtures/webm_chunks/webm_decoded.npy", "rb") as f:
        expected_data = np.load(f)

    p = Process(target=adding_chunks, args=(queue, stop_event))
    p.start()
    try:
        current_frame = 0
        with sut as stream:
            while current_frame < expected_data.shape[0]:
                actual_chunk = next(stream)
                chunk_size = actual_chunk.shape[0]
                assert np.allclose(actual_chunk, expected_data[current_frame : current_frame + chunk_size])
                current_frame += chunk_size
            stream.stop()
        assert current_frame == expected_data.shape[0]
    finally:
        p.join()
