from marqo.s2_inference.errors import VectoriseError
from marqo.s2_inference.sbert_utils import Model
from marqo.s2_inference.models.model_type import ModelType


class NO_MODEL(Model):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def load(self, *args, **kwargs) -> None:
        pass

    def encode(self, *args, **kwargs) -> None:
        raise VectoriseError(f"Cannot vectorise anything with '{ModelType.NoModel}' "
                             f"This model is intended for adding documents and searching with custom vectors only. "
                             f"If vectorisation is needed, please use a different model.")