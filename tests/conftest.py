import os
from collections.abc import Generator

import pytest

from ols2t.models import FileStream

test_dir = os.path.abspath(os.path.dirname(__file__))
fixture_dir = os.path.join(test_dir, "fixtures")


@pytest.fixture
def hello_fixture() -> Generator[FileStream, None, None]:
    yield FileStream(path=os.path.join(fixture_dir, "hello_ja.wav"))
