from marqo.inference.native_inference.content_preprocessing import split_prefix_preprocess_text, \
    download_and_preprocess_image
from marqo.inference.native_inference.embedding_models.open_clip_model import OPEN_CLIP
from marqo.inference.native_inference.inference_pipeline.abstract_inference_pipeline import AbstractInferencePipeline
from marqo.inference.type import *

OpenCLIPPreprocessedContent = Union[InferenceErrorModel, List[Tuple[str, Tensor]]]


class OpenCLIPInferencePipeline(AbstractInferencePipeline):

    MAX_BATCH_SIZE = 16

    def __init__(self, model: OPEN_CLIP, inference_request: InferenceRequest):
        super().__init__(model = model, inference_request = inference_request)


    def run_pipeline(self) -> InferenceResult:
        preprocessed_content_list: List[OpenCLIPPreprocessedContent] = self._content_preprocessing()

        embeddings: List[ndarray] = self._encode_processed_content(preprocessed_content_list)

        formated_result: InferenceResult = self.format_results(preprocessed_content_list, embeddings)
        return formated_result

    def _content_preprocessing(self) -> List[OpenCLIPPreprocessedContent]:
        if self.inference_request.modality == Modality.TEXT:
            results = split_prefix_preprocess_text(
                self.inference_request.contents,
                self.model.get_preprocessor(),
                self.inference_request.preprocessing_config
            )
        elif self.inference_request.modality == Modality.IMAGE:
            results = download_and_preprocess_image(
                self.inference_request.contents,
                self.model.get_preprocessor(),
                self.inference_request.preprocessing_config,
                self.inference_request.return_individual_error
            )
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        return results

    def _encode_processed_content(self, preprocessed_content_list: List[OpenCLIPPreprocessedContent]) -> List[ndarray]:
        flattened_content: List[Tensor] = self._collect_tensors(preprocessed_content_list)
        if len(flattened_content) > 0:
            embeddings = []
            for i in range(0, len(flattened_content), self.MAX_BATCH_SIZE):
                batch: List[Tensor] = flattened_content[i:i + self.MAX_BATCH_SIZE]
                batch_embeddings: List[ndarray] = self.model.encode(
                    inputs = batch,
                    modality = self.inference_request.modality,
                    normalize = self.inference_request.model_config.normalize_embeddings)
                embeddings.extend([embeddings for embeddings in batch_embeddings])

            if len(embeddings) != len(flattened_content):
                raise ValueError("The number of embeddings does not match the number of contents")

            return embeddings
        else:
            return []

    def _collect_tensors(self, preprocessed_content: list[list[tuple[str, Tensor]]]) -> list[Tensor]:
        collected_tensors = []
        for chunk in preprocessed_content:
            if isinstance(chunk, list):
                for _, tensor in chunk:
                    if isinstance(tensor, Tensor):
                        collected_tensors.append(tensor)
                    else:
                        raise ValueError(f"Expected tensor but got {type(tensor)}")
            elif isinstance(chunk, (MediaDownloadError, PreprocessingError)):
                continue
        return collected_tensors


