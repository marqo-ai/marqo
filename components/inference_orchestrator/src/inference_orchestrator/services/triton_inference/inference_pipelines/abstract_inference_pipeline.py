from abc import ABC, abstractmethod
from typing import Union

from numpy import ndarray

from inference_orchestrator.schemas.api import (
    AudioPreprocessingConfig,
    ImagePreprocessingConfig,
    Modality,
    TextPreprocessingConfig,
    VideoPreprocessingConfig,
)
from inference_orchestrator.schemas.api import InferenceErrorModel, InferenceResult
from inference_orchestrator.services.media_download_and_preprocess.media_download_and_preprocess import (
    process_batch,
)
from inference_orchestrator.services.media_download_and_preprocess.split_text import (
    prefix_text_chunks,
    split_text,
)
from inference_orchestrator.services.triton_inference.embedding_models.abstract_preprocessor import (
    AbstractPreprocessor,
)

PreprocessedContent = list[tuple[str, Union[str, any]]]


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
    def format_results(
        preprocessed_content_list: list, embeddings: list[ndarray]
    ) -> InferenceResult:
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
            raise ValueError(
                "The formatted results length does not match the input content length"
            )
        return InferenceResult(result=results)

    @staticmethod
    def split_prefix_preprocess_text(
        content: list[str],
        preprocessor: AbstractPreprocessor,
        preprocessing_config: TextPreprocessingConfig,
    ) -> list[PreprocessedContent]:
        """
        The function that handles the chunking(splitting), prefixing, and preprocessing of text content.
        Args:
            content: the content to be chunked, downloaded, and preprocessed.
            preprocessor: the preprocessor to be used for preprocessing the content.
            preprocessing_config: the preprocessing configuration to be used for preprocessing the content. This
                includes the text splitting configuration, text prefix, and chunking configuration for audio and video.

        Returns:
            Results in the form of a list[list[tuple[str, Any]]], where Any depends on the preprocessor used.

        Examples:
            E.g., 1
                The input is ["This is a test sentence", "Test"] with the text prefix "prefix: ", and split by word, and the
                preprocessor is a text preprocessor that returns tensors.

                The output will be
                    [
                        [("This is a", tensor), ("a test sentence", tensor)], # 2 chunks for the first content
                        [("Test", tensor)] # 1 chunk for the second content
                    ]
            E.g., 2
                The input is ["This is a test sentence", "Test"] with the text prefix "prefix: ", and split by word, and the
                preprocessor is a text preprocessor from HuggingFace that returns strings.

                The output will be
                    [
                        [
                            ("This is a", "prefix: this is a"),
                            ("a test sentence", "prefix: a test sentence")
                        ], # 2 chunks for the first content
                        [
                            ("Test", "prefix: Test")
                        ] # 1 chunk for the second content
                    ]

            IMPORTANT: That the tensor is generated with prefix, while the chunk does not contain the prefix.
        """

        def apply_prefix(text_list: list[str]) -> list[str]:
            if preprocessing_config.text_prefix is not None:
                return prefix_text_chunks(text_list, preprocessing_config.text_prefix)
            return text_list

        results: list[PreprocessedContent] = []

        # If chunking is enabled
        if preprocessing_config.should_chunk:
            for text in content:
                # Split the text into chunks
                chunks = split_text(
                    text,
                    split_by=preprocessing_config.chunk_config.split_method,
                    split_length=preprocessing_config.chunk_config.split_length,
                    split_overlap=preprocessing_config.chunk_config.split_overlap,
                )
                raw_content = chunks.copy()

                # Apply prefix if needed
                prefixed_chunks = apply_prefix(chunks)

                # Preprocess chunks
                preprocessed_chunks = preprocessor.preprocess(
                    prefixed_chunks, Modality.TEXT
                )

                # Validation
                if len(prefixed_chunks) != len(preprocessed_chunks):
                    raise ValueError(
                        "The number of preprocessed texts does not match the number of chunks"
                    )

                # Collect paired results
                results.append(list(zip(raw_content, preprocessed_chunks)))

        # If no chunking
        else:
            raw_content = content.copy()
            prefixed_content = apply_prefix(content)

            preprocessed_content = preprocessor.preprocess(
                prefixed_content, Modality.TEXT
            )

            # Pair each raw input with its processed output
            results = [
                [(raw_content[i], preprocessed_content[i])] for i in range(len(content))
            ]

        return results

    @staticmethod
    def download_and_preprocess_media(
        content: list[str],
        preprocessor: AbstractPreprocessor,
        preprocessing_config: Union[
            ImagePreprocessingConfig, AudioPreprocessingConfig, VideoPreprocessingConfig
        ],
        return_individual_error: bool = True,
    ) -> list[PreprocessedContent]:
        results = process_batch(
            content=content,
            preprocessor=preprocessor,
            preprocessing_config=preprocessing_config,
            return_individual_error=return_individual_error,
        )
        return results
