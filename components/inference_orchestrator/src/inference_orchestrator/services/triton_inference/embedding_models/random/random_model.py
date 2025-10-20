from typing import List

import blake3
import numpy as np
from numpy import ndarray

from inference_orchestrator.schemas.api import Modality
from inference_orchestrator.services.triton_inference.embedding_models.abstract_embedding_model import (
    AbstractEmbeddingModel,
)
from inference_orchestrator.services.triton_inference.embedding_models.abstract_preprocessor import (
    AbstractPreprocessor,
)
from inference_orchestrator.services.triton_inference.embedding_models.random.random_model_properties import (
    RandomModelProperties,
)


class RandomModelPreprocessor(AbstractPreprocessor):
    def __init__(self) -> None:
        super().__init__()

    def preprocess(self, inputs: list[str], modality) -> list[str]:
        """No preprocessing is done for the random model"""
        return inputs


class RandomModel(AbstractEmbeddingModel):
    """
    A model that generates random embeddings based on a hash of the input string.

    This model is primarily for testing and demonstration purposes. It does not require any external dependencies such
    as model management or Triton clients.
    """

    def __init__(
        self, model_properties: dict, *args, **kwargs
    ) -> None:  # Drop unused args, kwargs
        super().__init__(
            model_properties, model_management_client=None, triton_client=None
        )

        self._model_properties = self._build_model_properties()
        self.preprocessor = RandomModelPreprocessor()

    def _load_necessary_components(self):
        pass

    def _check_loaded_components(self):
        pass

    def load(self):
        pass

    def unload(self, remove_model: bool = False):
        pass

    def _build_model_properties(self) -> RandomModelProperties:
        return RandomModelProperties(**self.model_properties)

    def _get_seed_from_string(self, content: str) -> int:
        """Creates a deterministic seed from the input string."""
        h = blake3.blake3(content.encode("utf-8")).hexdigest()
        return int(h[:8], 16)  # 32-bit seed, like your MD5 version

    def encode(
        self, inputs: List[str], modality: Modality, normalize: bool = True
    ) -> List[ndarray]:
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
