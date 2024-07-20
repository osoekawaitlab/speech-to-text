from io import BufferedReader

import numpy as np
from numpy.typing import NDArray

def decode_audio(fp: BufferedReader) -> NDArray[np.float32]: ...
