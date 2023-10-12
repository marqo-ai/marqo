# Implements a 'model' that is a placeholder for an index.
# No vectorisation can be done for this model.
# Raises an error if any vectorisation is attempted.
# To be used for custom vectors in add docs and search only.

import functools
from marqo.s2_inference.sbert_utils import Model
from marqo.s2_inference.types import Union, FloatTensor, List, ndarray, ImageType
from marqo.tensor_search.enums import SpecialModels
from marqo.s2_inference import errors

class NO_MODEL(Model):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def load(self) -> None:
        self.model = None

    def encode(self, sentence: Union[str, List[str]], normalize: bool = True, **kwargs) -> Union[FloatTensor, ndarray]:
        raise errors.IllegalVectoriseError(f"Cannot vectorise anything with {SpecialModels.no_model}. This model is intended for "
                                           f"adding documents and searching with custom vectors only. If vectorisation is needed, "
                                           f"please use a different model.")