from marqo.inference.native_inference.content_preprocessing import split_prefix_preprocess_text
from marqo.inference.native_inference.embedding_models.random_model import RandomModel
from marqo.inference.native_inference.inference_pipeline.abstract_inference_pipeline import AbstractInferencePipeline
from marqo.inference.type import *

RandomModelPreprocessedContent = Union[InferenceErrorModel, List[Tuple[str, str]]]


class RandomModelInferencePipeline(AbstractInferencePipeline):

    MAX_BATCH_SIZE = 128

    def __init__(self, model: RandomModel, inference_request: InferenceRequest):
        super().__init__(model = model, inference_request = inference_request)

    def run_pipeline(self) -> InferenceResult:
        preprocessed_content_list: List[RandomModelPreprocessedContent] = self._content_preprocessing()
        embeddings: List[ndarray] = self._encode_processed_content(preprocessed_content_list)
        formated_result: InferenceResult = self.format_results(preprocessed_content_list, embeddings)
        return formated_result

    def _content_preprocessing(self) -> List[RandomModelPreprocessedContent]:
        """
        Preprocess the content based on the modality.

        Returns:
            List[RandomModelPreprocessedContent]: The preprocessed content.
        """
        if self.inference_request.modality == Modality.TEXT:
            results = split_prefix_preprocess_text(
                self.inference_request.contents,
                self.model.get_preprocessor(),
                self.inference_request.preprocessing_config
            )
        elif self.inference_request.modality == Modality.IMAGE:
            return [[(content, content), ] for content in self.inference_request.contents]
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        return results

    def _encode_processed_content(self, preprocessed_content_list: List[RandomModelPreprocessedContent]) -> List[ndarray]:
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

    def _collect_valid_content_to_encode(self, preprocessed_content: list[RandomModelPreprocessedContent]) -> list[str]:
        """
        Collect the valid content to encode from the preprocessed content. Each individual content can be
        an InferenceError, or a list of tuples with the original text and the preprocessed content. The
        valid content to encode in this model is string.

        Args:
            preprocessed_content: A list of preprocessed content.

        Returns:
            list[str]: A list of valid content to encode.

        Raises:
            ValueError: If the content is not a tensor, nor an InferenceError. This means there is an
            unexpected content type.
        """
        valid_content_to_encode = []
        valid_content_to_encode_type = (str, )

        for chunk in preprocessed_content:
            if isinstance(chunk, list):
                for _, content_to_encode in chunk:
                    if isinstance(content_to_encode, valid_content_to_encode_type):
                        valid_content_to_encode.append(content_to_encode)
                    else:
                        raise ValueError(f"Expected {valid_content_to_encode_type} "
                                         f"but got {type(content_to_encode)}")
            elif isinstance(chunk, InferenceErrorModel):
                continue
            else:
                raise ValueError(f"Unexpected content type: {type(chunk)}. "
                                 f"Should be a list of tuples or an InferenceError")
        return valid_content_to_encode