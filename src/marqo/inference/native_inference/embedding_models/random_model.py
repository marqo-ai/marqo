from typing import Optional

import numpy as np

from marqo.core.inference.api.inference import ModelAuth
from marqo.inference.native_inference.embedding_models.abstract_embedding_model import AbstractEmbeddingModel
from marqo.inference.native_inference.embedding_models.abstract_preprocessor import AbstractPreprocessor
from marqo.inference.native_inference.embedding_models.random_model_properties import RandomModelProperties
from marqo.s2_inference.types import List, ndarray, Modality
import hashlib

class RandomModelPreprocessor(AbstractPreprocessor):
    def __init__(self) -> None:
        super().__init__()

    def preprocess(self, inputs: list[str], modality) -> list[str]:
        """No preprocessing is done for the random model"""
        return inputs


class RandomModel(AbstractEmbeddingModel):

    def __init__(self, model_properties: dict, device: str, model_auth: Optional[ModelAuth] = None) -> None:
        super().__init__(model_properties, device, model_auth)

        self._model_properties = self._build_model_properties()
        self.preprocessor = RandomModelPreprocessor()

    def _load_necessary_components(self):
        pass

    def _check_loaded_components(self):
        pass

    def load(self):
        pass

    def _build_model_properties(self) -> RandomModelProperties:
        return RandomModelProperties(**self.model_properties)

    def _get_seed_from_string(self, content: str) -> int:
        """Creates a deterministic seed from the input string."""
        hash_object = hashlib.md5(content.encode('utf-8'))
        hash_digest = hash_object.hexdigest()
        return int(hash_digest[:8], 16)

    def encode(self, inputs: List[str], modality: Modality, normalize: bool = True) -> List[ndarray]:
        """
        Generate embeddings for the given inputs.

        The same content will always generate the same embedding.
        """
        embeddings = []

        for input_str in inputs:
            seed = self._get_seed_from_string(input_str)
            rng = np.random.default_rng(seed)

            # Generate embedding deterministically from seeded RNG
            embedding = rng.normal(size=self._model_properties.dimensions)
            if normalize:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

            embeddings.append(embedding)
        return embeddings

    def get_preprocessor(self) -> RandomModelPreprocessor:
        return self.preprocessor