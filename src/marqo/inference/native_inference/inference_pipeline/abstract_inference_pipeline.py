from abc import ABC, abstractmethod
from numpy import ndarray
from marqo.core.inference.api import *


class AbstractInferencePipeline(ABC):

    def __init__(self, model, inference_request):
        self.model = model
        self.inference_request = inference_request

    @abstractmethod
    def run_pipeline(self) -> InferenceResult:
        """
        The main method to run the inference pipeline.

        Returns:
            InferenceResult: The result of the inference pipeline.

        """
        pass

    @staticmethod
    def format_results(preprocessed_content_list: list, embeddings: list[ndarray]) -> InferenceResult:
        """
        Format the results of the inference pipeline into a InferenceResult object.
        Args:
            preprocessed_content_list: A list of preprocessed content to be formatted into InferenceResult.
            embeddings: A list of embeddings to be formatted into InferenceResult.
        Returns:
            InferenceResult: The formatted results.
        """
        results = []
        embedding_index = 0
        for chunk in preprocessed_content_list:
            chunk_results = []
            if isinstance(chunk, InferenceErrorModel):
                results.append(chunk)
                continue
            elif isinstance(chunk, list):
                for original_text, chunk_content in chunk:
                    chunk_results.append((original_text, embeddings[embedding_index]))
                    embedding_index += 1
            else:
                raise ValueError(f"Invalid chunk type: {type} for chunk: {chunk}")
            results.append(chunk_results)
        if len(results) != len(preprocessed_content_list):
            raise ValueError("The formatted results length does not match the input content length")
        return InferenceResult(result=results)