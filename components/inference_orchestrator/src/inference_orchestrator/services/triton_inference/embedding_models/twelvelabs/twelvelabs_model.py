import os
from typing import List

import numpy as np
from numpy import ndarray

from inference_orchestrator.core.logging import get_logger
from inference_orchestrator.schemas.api import Modality
from inference_orchestrator.services.triton_inference.embedding_models.abstract_embedding_model import (
    AbstractEmbeddingModel,
)
from inference_orchestrator.services.triton_inference.embedding_models.abstract_preprocessor import (
    AbstractPreprocessor,
)
from inference_orchestrator.services.triton_inference.embedding_models.twelvelabs.twelvelabs_model_properties import (
    TwelveLabsModelProperties,
)

logger = get_logger(__name__)

TWELVELABS_API_KEY_ENV_VAR = "TWELVELABS_API_KEY"


class TwelveLabsModelPreprocessor(AbstractPreprocessor):
    def preprocess(self, inputs: list, modality: Modality) -> list:
        """The TwelveLabs API ingests text and media URLs directly, so no
        client side preprocessing is required."""
        return inputs


class TwelveLabsModel(AbstractEmbeddingModel):
    """TwelveLabs Marengo multimodal embedding model.

    Embeddings are produced by the TwelveLabs API rather than by a locally
    served Triton model. Text and image content are embedded synchronously via
    ``embed.create``; video content is embedded via the asynchronous
    ``embed.tasks`` endpoint. All modalities share the same 512 dimensional
    embedding space, which makes Marengo a drop-in multimodal alternative to
    CLIP for cross-modal (e.g. text-to-video) retrieval.

    The API key is read from the ``TWELVELABS_API_KEY`` environment variable.
    Get a free key with a generous free tier at https://twelvelabs.io .
    """

    def __init__(self, model_properties: dict, *args, **kwargs) -> None:
        # Drop unused triton/model-management clients; Marengo is API served.
        super().__init__(
            model_properties, model_management_client=None, triton_client=None
        )
        self._model_properties = TwelveLabsModelProperties(**self.model_properties)
        self.preprocessor = TwelveLabsModelPreprocessor()
        self._client = None

    def _load_necessary_components(self):
        try:
            from twelvelabs import TwelveLabs
        except ImportError as e:  # pragma: no cover - exercised only without dep
            raise ImportError(
                "The `twelvelabs` package is required to use TwelveLabs Marengo "
                "models. Install it with `pip install twelvelabs`."
            ) from e

        api_key = os.environ.get(TWELVELABS_API_KEY_ENV_VAR)
        if not api_key:
            raise ValueError(
                f"The `{TWELVELABS_API_KEY_ENV_VAR}` environment variable must be "
                "set to use TwelveLabs Marengo models. Get a free key at "
                "https://twelvelabs.io ."
            )
        self._client = TwelveLabs(api_key=api_key)

    def _check_loaded_components(self):
        if self._client is None:
            raise RuntimeError("TwelveLabs client failed to initialise.")

    def encode(
        self, inputs: List[str], modality: Modality, normalize: bool = True
    ) -> List[ndarray]:
        """Embed each input string and return one (dimensions,) vector per input."""
        embeddings: List[ndarray] = []
        for content in inputs:
            vector = self._embed_one(content, modality)
            embedding = np.asarray(vector, dtype=np.float32)
            if normalize:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            embeddings.append(embedding)
        return embeddings

    def _embed_one(self, content: str, modality: Modality) -> List[float]:
        api_model_name = self._model_properties.api_model_name
        if modality == Modality.TEXT:
            response = self._client.embed.create(
                model_name=api_model_name, text=content
            )
            return self._first_segment(response.text_embedding, content)
        elif modality == Modality.IMAGE:
            response = self._client.embed.create(
                model_name=api_model_name, image_url=content
            )
            return self._first_segment(response.image_embedding, content)
        elif modality == Modality.VIDEO:
            task = self._client.embed.tasks.create(
                model_name=api_model_name, video_url=content
            )
            task = self._client.embed.tasks.wait_for_done(task_id=task.id)
            return self._first_segment(getattr(task, "video_embedding", None), content)
        else:
            raise ValueError(f"Unsupported modality for TwelveLabs model: {modality}")

    @staticmethod
    def _first_segment(embedding_result, content: str) -> List[float]:
        segments = (
            getattr(embedding_result, "segments", None) if embedding_result else None
        )
        if not segments:
            raise RuntimeError(
                f"TwelveLabs returned no embedding segments for content: {content!r}"
            )
        return segments[0].float_

    def get_preprocessor(self) -> TwelveLabsModelPreprocessor:
        return self.preprocessor

    def load(self):
        self._load_necessary_components()
        self._check_loaded_components()

    def unload(self, remove_files: bool = False):
        self._client = None
