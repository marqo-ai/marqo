from torch import Tensor
from typing import List, Tuple, Union

from marqo.core.inference.api import *

PreprocessedContent = Union[MediaDownloadError, PreprocessingError, List[Tuple[str, Tensor]]]
