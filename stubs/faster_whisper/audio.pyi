from io import BufferedReader

import numpy as np
from numpy.typing import NDArray

def decode_audio(fp: BufferedReader, sampling_rate: int) -> NDArray[np.float32]: ...
