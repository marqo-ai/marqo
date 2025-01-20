from torch import FloatTensor, Tensor
from PIL.Image import Image as ImageType
from numpy import ndarray
from enum import Enum
from typing import (
    Any, 
    Dict, 
    List, 
    Optional, 
    Union, 
    Callable,
    Tuple, 
    Iterable, 
    Type, 
    Literal
    )

class Modality(str, Enum):
    """language/TEXT is widely used by language models so we keep it as internal use for now.
    """
    TEXT = "language"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"