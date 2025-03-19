from PIL.Image import Image

from marqo.inference.native_inference.chunk_download_preprocess_content import chunk_download_preprocess_content
from marqo.inference.native_inference.embedding_models.random_model import RandomModel
from marqo.inference.native_inference.encode_content import encode_processed_content, format_results
from marqo.inference.native_inference.inference_pipeline.abstract_inference_pipeline import AbstractInferencePipeline
from marqo.inference.type import *

RandomModelPreprocessedContent = Union[InferenceErrorModel, List[Tuple[str, Union[str, Image]]]]


class RandomModelInferencePipeline(AbstractInferencePipeline):
    def __init__(self, model: RandomModel, inference_request: InferenceRequest):
        super().__init__(model = model, inference_request = inference_request)

    def run_pipeline(self) -> InferenceResult:
        preprocessed_content_list: List[RandomModelPreprocessedContent] = (
            self._chunk_download_preprocess_content())
        embeddings: List[ndarray] = self._encode_processed_content(preprocessed_content_list)
        formated_result: InferenceResult = format_results(preprocessed_content_list, embeddings)
        return formated_result

    def _chunk_download_preprocess_content(self) -> List[RandomModelPreprocessedContent]:
        return chunk_download_preprocess_content(
            content=self.inference_request.contents,
            modality=self.inference_request.modality,
            preprocessing_config=self.inference_request.preprocessing_config,
            preprocessor=self.model.get_preprocessor(),
        )

    def _encode_processed_content(self, preprocessed_content_list) -> List[ndarray]:
        return encode_processed_content(
            model=self.model,
            preprocessed_content_list=preprocessed_content_list,
            modality=self.inference_request.modality,
            normalize=self.inference_request.model_config.normalize_embeddings,
            maximum_batch_size=16
        )