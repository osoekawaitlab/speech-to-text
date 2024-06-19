import re

import ols2t


def test_version_is_exported() -> None:
    assert re.match(r"\d+\.\d+\.\d+", ols2t.__version__)
