from typing import Optional

import numpy as np

from marqo.core.inference.api import Inference, InferenceResult, InferenceRequest, InferenceError, InferenceErrorModel


class DummyInference(Inference):
    def __init__(self, model_dimension: int = 512, chunks: int = 2, error_on_prefix: Optional[str] = None):
        self.model_dimension = model_dimension
        self.chunks = chunks
        self.error_on_prefix = error_on_prefix

    def vectorise(self, request: InferenceRequest) -> InferenceResult:
        results = []
        for content in request.contents:
            try:
                if self.error_on_prefix and content.startswith(self.error_on_prefix):
                    raise Exception(content)
                if request.preprocessing_config.should_chunk:
                    results.append([(f'chunk_{i}', np.random.rand(self.model_dimension)) for i in range(self.chunks)])
                else:
                    results.append([(content, np.random.rand(self.model_dimension))])
            except Exception as e:
                error_message = f"Error processing content: {content}. Error: {str(e)}"
                if request.return_individual_error:
                    # If an error occurs for this specific content, add an InferenceError.
                    results.append(InferenceErrorModel(error_message=error_message))
                else:
                    raise InferenceError(error_message)

        return InferenceResult(result=results)