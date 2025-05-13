from typing import Tuple, List

import numpy as np

from marqo.core.inference.api import Inference, InferenceRequest, InferenceResult, InferenceError
from marqo.inference.inference_cache.marqo_inference_cache import MarqoInferenceCache


class CachingInference(Inference):
    def __init__(self, delegate: Inference, cache_size: int, cache_type: str):
        self.delegate = delegate
        self.inference_cache = MarqoInferenceCache(cache_size=cache_size, cache_type=cache_type,
                                                   value_size_lambda=lambda v: v.nbytes)

    def vectorise(self, request: InferenceRequest) -> InferenceResult:
        if not request.use_inference_cache:
            return self.delegate.vectorise(request)

        # TODO add comments
        device = request.device or 'none'
        model_name = request.model_config.model_name
        cache_key = f'{device}||{model_name}||{request.modality.value}'

        cached_result: List[Tuple[int, str, np.ndarray]] = []
        contents_to_vectorise: List[str] = []

        for index, content in enumerate(request.contents):
            embedding = self.inference_cache.get(cache_key, content)
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
                # TODO make sure the result is not chunked (only 1 tuple in the result)
                content, embedding = r[0]
                self.inference_cache.set(cache_key, content, embedding)

        # Merge result
        if cached_result:
            for loc, content, embedding in cached_result:
                inference_result.result.insert(loc, [(content, embedding)])

        return inference_result


