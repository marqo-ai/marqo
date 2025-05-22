from typing import Tuple, List

import numpy as np
import orjson
import hashlib

from marqo.core.inference.api import Inference, InferenceRequest, InferenceResult, InferenceError, Modality
from marqo.inference.inference_cache.marqo_inference_cache import MarqoInferenceCache


class CachingInference(Inference):
    def __init__(self, delegate: Inference, cache_size: int, cache_type: str):
        self.delegate = delegate
        self.inference_cache = MarqoInferenceCache(cache_size=cache_size, cache_type=cache_type)

    def vectorise(self, request: InferenceRequest) -> InferenceResult:
        if self.should_skip_cache(request):
            return self.delegate.vectorise(request)

        model_cache_key = self.model_cache_key(request.model_config.model_properties)

        cached_result: List[Tuple[int, str, np.ndarray]] = []
        contents_to_vectorise: List[str] = []

        for index, content in enumerate(request.contents):
            embedding = self.inference_cache.get(model_cache_key, content)
            if embedding is not None:
                cached_result.append((index, content, embedding))
            else:
                contents_to_vectorise.append(content)

        if not contents_to_vectorise:
            return InferenceResult(result=[[(content, embedding)] for _, content, embedding in cached_result])

        new_request = request.copy(update={"contents": contents_to_vectorise})
        inference_result = self.delegate.vectorise(new_request)

        for r in inference_result.result:
            if not isinstance(r, InferenceError):
                if len(r) > 1:
                    raise RuntimeError(f"Inference cache does not support chunking but got {len(r)} chunks. "
                                       f"Preprocessing config: "
                                       f"{orjson.dumps(dict(new_request.preprocessing_config)).decode('utf-8')}")
                content, embedding = r[0]
                self.inference_cache.set(model_cache_key, content, embedding)

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
        data = orjson.dumps(
            model_properties,
            option=orjson.OPT_SORT_KEYS
        )

        h = hashlib.new('md5')
        h.update(data)
        return h.hexdigest()

    def should_skip_cache(self, request):
        return (
            not request.use_inference_cache
            or request.device  # device is only specified to debug embedding, skip caching
            or request.modality != Modality.TEXT  # we only support text modality for now
            or request.preprocessing_config.should_chunk  # we do not support caching chunks
        )


