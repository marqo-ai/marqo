import hashlib
from typing import List, Optional, Tuple

import blake3
import numpy as np
import orjson

from inference_orchestrator.schemas.api import (
    Inference,
    InferenceErrorModel,
    InferenceRequest,
    InferenceResult,
    Modality,
)
from inference_orchestrator.services.inference_cache.marqo_inference_cache import (
    MarqoInferenceCache,
)


def is_base64_image(content: str) -> bool:
    return content.startswith("data:image/")


class CachingInference(Inference):
    def __init__(self, delegate: Inference, cache_size: int, cache_type: str):
        self.delegate = delegate
        self.inference_cache = MarqoInferenceCache(
            cache_size=cache_size, cache_type=cache_type
        )

    def vectorise(self, request: InferenceRequest) -> InferenceResult:
        if self.should_skip_cache(request):
            return self.delegate.vectorise(request)

        model_cache_key = self.model_cache_key(
            request.embedding_model_config.model_properties
        )

        cached_result: List[Tuple[int, str, np.ndarray]] = []
        contents_to_vectorise: List[str] = []

        for index, content in enumerate(request.contents):
            content_cache_key = self.content_cache_key(content, request.modality)
            if not content_cache_key:
                contents_to_vectorise.append(content)
                continue

            embedding = self.inference_cache.get(model_cache_key, content_cache_key)
            if embedding is not None:
                cached_result.append((index, content, embedding))
            else:
                contents_to_vectorise.append(content)

        if not contents_to_vectorise:
            return InferenceResult(
                result=[
                    [(content, embedding)] for _, content, embedding in cached_result
                ]
            )

        new_request = request.model_copy(update={"contents": contents_to_vectorise})
        inference_result = self.delegate.vectorise(new_request)

        for r in inference_result.result:
            if not isinstance(r, InferenceErrorModel):
                if len(r) > 1:
                    raise RuntimeError(
                        f"Inference cache does not support chunking but got {len(r)} chunks. "
                        f"Preprocessing config: "
                        f"{orjson.dumps(dict(new_request.preprocessing_config)).decode('utf-8')}"
                    )
                content, embedding = r[0]
                content_cache_key = self.content_cache_key(content, request.modality)
                if content_cache_key:
                    self.inference_cache.set(
                        model_cache_key, content_cache_key, embedding
                    )

        # Merge result
        if cached_result:
            for loc, content, embedding in cached_result:
                inference_result.result.insert(loc, [(content, embedding)])

        return inference_result

    def model_cache_key(self, model_properties) -> str:
        """
        Generate a md5 hash (32 bytes) based on the modal_properties dictionary. Since we need to store the model
        properties as part of the key in the cache, we hash the dumped json to get a smaller value to save the memory
        space used by cache. In most use cases, there's only one model, md5 is good enough to avoid collision
        """
        data = orjson.dumps(model_properties, option=orjson.OPT_SORT_KEYS)
        return hashlib.md5(data).hexdigest()

    def content_cache_key(self, content: str, modality: Modality) -> Optional[str]:
        """
        Generate appropriate cache key for content based on modality.

        For TEXT modality: use content directly
        For IMAGE modality:
            - if base64 image: use blake3 hash with prefix
            - otherwise: use content directly (will be skipped in caching logic)

        Args:
            content: The content string
            modality: The modality type

        Returns:
            Cache key string, None if it should not be cached
        """
        if modality == Modality.TEXT:
            # Use original content for text and non-base64 images
            return content
        elif modality == Modality.IMAGE and is_base64_image(content):
            # Use blake3 hash for base64 images to save memory
            hash_digest = blake3.blake3(content.encode()).hexdigest()
            return f"blake3:{hash_digest}"
        else:
            # should not cache non-base64-encoded images
            return None

    def should_skip_cache(self, request):
        return (
            not request.use_inference_cache
            or request.modality
            not in [
                Modality.TEXT,
                Modality.IMAGE,
            ]  # we support text and image modalities
            or request.preprocessing_config.should_chunk  # we do not support caching chunks
        )
