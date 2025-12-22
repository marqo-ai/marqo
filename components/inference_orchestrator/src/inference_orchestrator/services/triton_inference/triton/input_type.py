from dataclasses import dataclass
from enum import Enum

import numpy as np


@dataclass(frozen=True)
class InputType:
    code: str
    dtype: type


class InputTypeEnum(Enum):
    INT32 = InputType("INT32", np.int32)
    INT64 = InputType("INT64", np.int64)
    FP32 = InputType("FP32", np.float32)
    FP16 = InputType("FP16", np.float16)
    UINT8 = InputType("UINT8", np.uint8)
