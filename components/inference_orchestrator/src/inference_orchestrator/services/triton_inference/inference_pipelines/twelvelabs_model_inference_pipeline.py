from typing import List, Tuple, Union

from numpy import ndarray

from inference_orchestrator.schemas.api import (
    InferenceErrorModel,
    InferenceRequest,
    InferenceResult,
    Modality,
)
from inference_orchestrator.services.triton_inference.content_preprocessing import (
    split_prefix_preprocess_text,
)
from inference_orchestrator.services.triton_inference.embedding_models.twelvelabs.twelvelabs_model import (
    TwelveLabsModel,
)
from inference_orchestrator.services.triton_inference.inference_pipelines.abstract_inference_pipeline import (
    AbstractInferencePipeline,
)

TwelveLabsModelPreprocessedContent = Union[InferenceErrorModel, List[Tuple[str, str]]]


class TwelveLabsModelInferencePipeline(AbstractInferencePipeline):
    """Inference pipeline for TwelveLabs Marengo models.

    Text content is chunked/prefixed like other text pipelines; image and video
    content (URLs) are passed straight through to the TwelveLabs API. Requests
    are sent one item at a time because Marengo's embed endpoint embeds a single
    piece of content per call.
    """

    # Marengo's embed endpoint handles one content item per call; this only
    # bounds how many we slice at a time, not a server side batch size.
    MAX_BATCH_SIZE = 8

    def __init__(self, model: TwelveLabsModel, inference_request: InferenceRequest):
        super().__init__(model=model, inference_request=inference_request)

    def run_pipeline(self) -> InferenceResult:
        preprocessed_content_list: List[TwelveLabsModelPreprocessedContent] = (
            self._content_preprocessing()
        )
        embeddings: List[ndarray] = self._encode_processed_content(
            preprocessed_content_list
        )
        return self.format_results(preprocessed_content_list, embeddings)

    def _content_preprocessing(self) -> List[TwelveLabsModelPreprocessedContent]:
        if self.inference_request.modality == Modality.TEXT:
            return split_prefix_preprocess_text(
                self.inference_request.contents,
                self.model.get_preprocessor(),
                self.inference_request.preprocessing_config,
            )
        elif self.inference_request.modality in (Modality.IMAGE, Modality.VIDEO):
            return [[(content, content)] for content in self.inference_request.contents]
        else:
            raise ValueError(f"Unsupported modality: {self.inference_request.modality}")

    def _encode_processed_content(
        self, preprocessed_content_list: List[TwelveLabsModelPreprocessedContent]
    ) -> List[ndarray]:
        content_to_encode: List[str] = self._collect_valid_content_to_encode(
            preprocessed_content_list
        )
        if not content_to_encode:
            return []

        embeddings: List[ndarray] = []
        for i in range(0, len(content_to_encode), self.MAX_BATCH_SIZE):
            batch: List[str] = content_to_encode[i : i + self.MAX_BATCH_SIZE]
            batch_embeddings: List[ndarray] = self.model.encode(
                inputs=batch,
                modality=self.inference_request.modality,
                normalize=self.inference_request.embedding_model_config.normalize_embeddings,
            )
            embeddings.extend(batch_embeddings)

        if len(embeddings) != len(content_to_encode):
            raise ValueError(
                "The number of embeddings does not match the number of contents"
            )
        return embeddings

    def _collect_valid_content_to_encode(
        self, preprocessed_content: List[TwelveLabsModelPreprocessedContent]
    ) -> List[str]:
        valid_content_to_encode: List[str] = []
        for chunk in preprocessed_content:
            if isinstance(chunk, list):
                for _, content_to_encode in chunk:
                    if isinstance(content_to_encode, str):
                        valid_content_to_encode.append(content_to_encode)
                    else:
                        raise ValueError(
                            f"Expected (str,) but got {type(content_to_encode)}"
                        )
            elif isinstance(chunk, InferenceErrorModel):
                continue
            else:
                raise ValueError(
                    f"Unexpected content type: {type(chunk)}. "
                    f"Should be a list of tuples or an InferenceError"
                )
        return valid_content_to_encode
