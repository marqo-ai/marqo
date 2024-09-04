from marqo.core.inference.models.abstract_embedding_model import AbstractEmbeddingModel


class HuggingFaceModel(AbstractEmbeddingModel):
    """The concrete class for all sentence transformers models loaded from Hugging Face.



    """

    def __init__(self, model_name: str, ):




    def _check_loaded_components(self):

        pass



    def _load_necessary_components(self):
        if self.tokenizer is None:
            self.tokenizer = self._load_tokenizer()
        pass

