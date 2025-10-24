from torch import Tensor

from marqo.core.inference.api import *
from PIL.Image import Image

# The type of the preprocessed content is unknown as it is dependent on the model, so we leave any here.
# However, it is guaranteed to be a list of either InferenceErrorModel or the model-specific content.
PreprocessedContent = Union[InferenceErrorModel, List[Tuple[str, Any]]]
