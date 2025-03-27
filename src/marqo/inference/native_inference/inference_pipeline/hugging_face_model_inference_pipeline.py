from marqo.inference.native_inference.content_preprocessing import split_prefix_preprocess_text
from marqo.inference.native_inference.embedding_models.hugging_face_model import HuggingFaceModel
from marqo.inference.native_inference.inference_pipeline.abstract_inference_pipeline import AbstractInferencePipeline
from marqo.inference.type import *

HuggingFacePreprocessedContent = Union[InferenceErrorModel, List[Tuple[str, str]]]


class HuggingFaceModelInferencePipeline(AbstractInferencePipeline):
    """
    A class that handles the inference pipeline for HuggingFace models.

    This class is responsible for the content preprocessing, encoding, and formatting the results for HuggingFace models.

    Attributes:
        VALID_CONTENT_TO_ENCODE_TYPE (tuple): The valid content type to passed to the model.encode method,
            this is a model specific type. In this case, it is a string.
        MAX_BATCH_SIZE (int): The maximum batch size to encode the content.
    """
    VALID_CONTENT_TO_ENCODE_TYPE = (str, )
    MAX_BATCH_SIZE = 32

    def __init__(self, model: HuggingFaceModel, inference_request: InferenceRequest):
        super().__init__(model = model, inference_request = inference_request)


    def run_pipeline(self) -> InferenceResult:
        preprocessed_content_list: List[HuggingFacePreprocessedContent] = self._content_preprocessing()

        embeddings: List[ndarray] = self._encode_processed_content(preprocessed_content_list)

        formated_result: InferenceResult = self.format_results(preprocessed_content_list, embeddings)
        return formated_result

    def _content_preprocessing(self) -> List[HuggingFacePreprocessedContent]:
        """
        Preprocess the content based on the modality.

        If it is a text modality, the content will be split, prefixed, and preprocessed as required by the
        preprocessing_config.

        However, if it's an image, audio, or video modality, this normally means the content is a URL from the
        search request. In this case, we just use a default TextPreprocessingConfig to preprocess the content.

        Returns:
            List[OpenCLIPPreprocessedContent]: The preprocessed content.
        """
        if self.inference_request.modality == Modality.TEXT:
            results = split_prefix_preprocess_text(
                self.inference_request.contents,
                self.model.get_preprocessor(),
                self.inference_request.preprocessing_config
            )
        elif self.inference_request.modality in [Modality.IMAGE, Modality.AUDIO, Modality.VIDEO]:
            results = split_prefix_preprocess_text(
                self.inference_request.contents,
                self.model.get_preprocessor(),
                TextPreprocessingConfig() # Use a default TextPreprocessingConfig
            )
        else:
            raise ValueError(f"Unsupported modality: {self.inference_request.modality}")
        return results

    def _encode_processed_content(self, preprocessed_content_list: List[HuggingFacePreprocessedContent]) \
            -> List[ndarray]:
        """
        Encode the preprocessed content into embeddings.

        Args:
            preprocessed_content_list: A list of preprocessed content.

        Returns:
            List[ndarray]: The embeddings. Each embedding is a numpy array with (Dimension, ) shape.
        """
        content_to_encode: List[str] = self._collect_valid_content_to_encode(preprocessed_content_list)

        if not content_to_encode:
            return []

        embeddings: List[ndarray] = []
        for i in range(0, len(content_to_encode), self.MAX_BATCH_SIZE):
            batch: List[str] = content_to_encode[i:i + self.MAX_BATCH_SIZE]
            batch_embeddings: List[ndarray] = self.model.encode(
                inputs=batch,
                modality=self.inference_request.modality,
                normalize=self.inference_request.model_config.normalize_embeddings
            )
            embeddings.extend(batch_embeddings)

        if len(embeddings) != len(content_to_encode):
            raise ValueError("The number of embeddings does not match the number of contents")

        return embeddings

    def _collect_valid_content_to_encode(self, preprocessed_content: list[HuggingFacePreprocessedContent]) \
            -> list[str]:
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
                            f"Expected {self.VALID_CONTENT_TO_ENCODE_TYPE} but "
                            f"got {type(content_to_encode)}"
                        )
            elif isinstance(chunk, InferenceErrorModel):
                continue
            else:
                raise ValueError(f"Unexpected content type: {type(chunk)}. "
                                 f"Should be a list of tuples or an InferenceError")
        return valid_content_to_encode


