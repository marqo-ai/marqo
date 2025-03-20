"""
A module that handles the chunking, downloading, and preprocessing of content for inference.

The reason that we need to do these 3 steps together is that:
    1. For Audio and Video, we need to chunk the content before downloading it.
    2. For Text, we need to chunk the content before preprocessing it.
    3. For Image, we need to preprocess the content right after downloading it to avoid memory issues.
Thus, this module is responsible for handling the chunking, downloading, and preprocessing of content for native
inference.
"""

from marqo.inference.media_download_and_preprocess.media_download_and_preprocess import (
    process_batch)
from marqo.inference.native_inference.embedding_models.abstract_preprocessor import AbstractPreprocessor
from marqo.inference.media_download_and_preprocess.split_text import split_text, prefix_text_chunks
from marqo.inference.type import *


def split_prefix_preprocess_text(
        content: list[str],
        preprocessor: AbstractPreprocessor,
        preprocessing_config: TextPreprocessingConfig
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
                split_overlap=preprocessing_config.chunk_config.split_overlap
            )
            raw_content = chunks.copy()

            # Apply prefix if needed
            prefixed_chunks = apply_prefix(chunks)

            # Preprocess chunks
            preprocessed_chunks = preprocessor.preprocess(prefixed_chunks, Modality.TEXT)

            # Validation
            if len(prefixed_chunks) != len(preprocessed_chunks):
                raise ValueError("The number of preprocessed texts does not match the number of chunks")

            # Collect paired results
            results.append(list(zip(raw_content, preprocessed_chunks)))

    # If no chunking
    else:
        raw_content = content.copy()
        prefixed_content = apply_prefix(content)

        preprocessed_content = preprocessor.preprocess(prefixed_content, Modality.TEXT)

        # Pair each raw input with its processed output
        results = [[(raw_content[i], preprocessed_content[i])] for i in range(len(content))]

    return results


def download_and_preprocess_image(
        content: list[str], preprocessor: AbstractPreprocessor,
        preprocessing_config: ImagePreprocessingConfig, return_individual_error: bool = True) \
            -> list[PreprocessedContent]:

    results = process_batch(
        content=content,
        preprocessor=preprocessor,
        modality=Modality.IMAGE,
        thread_count=preprocessing_config.download_thread_count,
        media_download_headers=preprocessing_config.download_header,
        download_timeout_ms=preprocessing_config.download_timeout_ms,
        audio_video_preprocessing_config=None,
        return_individual_error=return_individual_error
    )
    return results