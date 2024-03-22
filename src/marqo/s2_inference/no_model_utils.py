from marqo.s2_inference.errors import VectoriseError
from marqo.s2_inference.sbert_utils import Model


class NoModel(Model):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def load(self, *args, **kwargs) -> None:
        pass

    def encode(self, *args, **kwargs) -> None:
        raise VectoriseError("'no_model' cannot vectorise your content. Please provide your own vectors "
                             "or use a different model")