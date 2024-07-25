from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray
from pytest_mock import MockerFixture

from ols2t.models import MicrophoneStream


def test_microphone_stream(
    mocker: MockerFixture,
    longtext_all_decoded_one_second_chunks: Iterable[NDArray[np.float32]],
    longtext_all_decoded_fixture_path: str,
) -> None:
    PyAudio = mocker.patch("ols2t.models.PyAudio")
    PyAudio.return_value.open.return_value.read.side_effect = longtext_all_decoded_one_second_chunks
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
