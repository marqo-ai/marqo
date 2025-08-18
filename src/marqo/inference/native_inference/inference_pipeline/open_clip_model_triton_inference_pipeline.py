import numpy as np
from sympy import content

from marqo.inference.native_inference.content_preprocessing import split_prefix_preprocess_text, \
    download_and_preprocess_media
from marqo.inference.native_inference.inference_pipeline.abstract_inference_pipeline import AbstractInferencePipeline
from marqo.inference.type import *
from marqo.inference.native_inference.content_preprocessing import split_prefix_preprocess_text, \
    download_and_preprocess_media
from marqo.inference.native_inference.embedding_models.open_clip_model import OpenCLIPModel
from marqo.inference.native_inference.inference_pipeline.abstract_inference_pipeline import AbstractInferencePipeline
from marqo.inference.type import *
import torch
from marqo.inference.triton_inference.triton_inference_client import TritonInferenceClient

OpenCLIPPreprocessedContent = Union[InferenceErrorModel, List[Tuple[str, Tensor]]]


class OpenCLIPModelTritonInferencePipeline(AbstractInferencePipeline):

    VALID_CONTENT_TO_ENCODE_TYPE = (Tensor,)
    MAX_BATCH_SIZE = 16

    def __init__(self, model: OpenCLIPModel, inference_request: InferenceRequest, triton_client: TritonInferenceClient):
        super().__init__(model = model, inference_request = inference_request)
        self.triton_client = triton_client
        self.model.model = None

    def run_pipeline(self) -> InferenceResult:
        preprocessed_content_list: List[OpenCLIPPreprocessedContent] = self._content_preprocessing()

        embeddings: List[ndarray] = self._encode_processed_content(preprocessed_content_list)

        formated_result: InferenceResult = self.format_results(preprocessed_content_list, embeddings)
        return formated_result

    def _content_preprocessing(self) -> List[OpenCLIPPreprocessedContent]:
        """
        Preprocess the content based on the modality.

        Returns:
            List[OpenCLIPPreprocessedContent]: The preprocessed content.
        """
        if self.inference_request.modality == Modality.TEXT:
            results = split_prefix_preprocess_text(
                self.inference_request.contents,
                self.model.get_preprocessor(),
                self.inference_request.preprocessing_config
            )
        elif self.inference_request.modality == Modality.IMAGE:
            results = download_and_preprocess_media(self.inference_request.contents, self.model.get_preprocessor(),
                                                    self.inference_request.preprocessing_config,
                                                    self.inference_request.return_individual_error)
        else:
            # TODO - Raise an unsupported modality error
            raise ValueError(f"Unsupported modality: {self.inference_request.modality}")
        return results

    def _encode_processed_content(self, preprocessed_content_list: List[OpenCLIPPreprocessedContent]) -> List[
        ndarray]:
        """
        Encode the preprocessed content into embeddings.

        Args:
            preprocessed_content_list: A list of preprocessed content.

        Returns:
            List[ndarray]: The embeddings. Each embedding is a numpy array with (Dimension, ) shape.
        """
        content_to_encode: List[Tensor] = self._collect_valid_content_to_encode(preprocessed_content_list)
        if not content_to_encode:
            return []

        content_to_encode: ndarray = torch.cat(content_to_encode, dim=0).cpu().to(torch.int32).numpy()

        raw_embeddings: ndarray = self.triton_client.encode(
            inputs=content_to_encode,
            modality=self.inference_request.modality
        )

        return [r for r in raw_embeddings]


    def _collect_valid_content_to_encode(self, preprocessed_content: list[OpenCLIPPreprocessedContent]) -> list[Tensor]:
        """
        Collect the valid content to encode from the preprocessed content. Each individual content can be
        an InferenceError, or a list of tuples with the original text and the preprocessed content. The
        valid content to encode in this model is Tensor.

        Args:
            preprocessed_content: A list of preprocessed content.

        Returns:
            list[Tensor]: A list of valid content to encode.

        Raises:
            ValueError: If the content is not a tensor, nor an InferenceError. This means there is an
            unexpected content type.
        """
        valid_content_to_encode = []

        for chunk in preprocessed_content:
            if isinstance(chunk, list):
                for _, content_to_encode in chunk:
                    if isinstance(content_to_encode, self.VALID_CONTENT_TO_ENCODE_TYPE):
                        valid_content_to_encode.append(content_to_encode)
                    else:
                        raise ValueError(
                            f"Expected {self.VALID_CONTENT_TO_ENCODE_TYPE} but got "
                            f"{type(content_to_encode)}"
                        )
            elif isinstance(chunk, InferenceErrorModel):
                continue
            else:
                raise ValueError(f"Unexpected content type: {type(chunk)}. "
                                 f"Should be a list of tuples or an InferenceError")
        return valid_content_to_encode


