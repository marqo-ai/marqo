from marqo.s2_inference.errors import VectoriseError
from marqo.s2_inference.sbert_utils import Model
from marqo.s2_inference.models.model_type import ModelType


class NO_MODEL(Model):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def load(self, *args, **kwargs) -> None:
        pass

    def encode(self, *args, **kwargs) -> None:
        raise VectoriseError(f"Cannot vectorise anything with '{ModelType.NO_MODEL}'. "
                             f"This model is intended for adding documents and searching with custom vectors only."
                             f"If searching using this index, you can set the search_method to LEXICAL for lexical search, "
                             f"or remove the field from tensor_fields when calling add_documents to make your content" 
                             f"only available for lexical search or filtering."
                             f"If vectorisation is needed, please re-create the index a different model.")
