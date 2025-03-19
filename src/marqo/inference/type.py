from torch import Tensor
from typing import List, Tuple, Union

from marqo.core.inference.api import *

PreprocessedContent = Union[InferenceErrorModel, List[Tuple[str, Tensor]]]
