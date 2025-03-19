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
        content: list[str], preprocessor: AbstractPreprocessor,
        preprocessing_config: TextPreprocessingConfig) -> list[PreprocessedContent]:
    results: list[PreprocessedContent] = []
    if preprocessing_config.should_chunk:
        for text in content:
            splitted_text: list[str] = split_text(
                text,
                split_by=preprocessing_config.chunk_config.split_method,
                split_length=preprocessing_config.chunk_config.split_length,
                split_overlap=preprocessing_config.chunk_config.split_overlap
            )
            if preprocessing_config.text_prefix is not None:
                splitted_text = prefix_text_chunks(splitted_text, preprocessing_config.text_prefix)
            preprocessed_text_list: list[Tensor] = preprocessor.preprocess(inputs=splitted_text, modality=Modality.TEXT)
            if len(splitted_text) != len(preprocessed_text_list):
                raise ValueError("The number of preprocessed text does not match the number of splitted text")
            results.append([(splitted_text[i], preprocessed_text_list[i]) for i in range(len(splitted_text))])
    else:
        if preprocessing_config.text_prefix is not None:
            content = prefix_text_chunks(content, preprocessing_config.text_prefix)
        preprocessed_text_list: list[Tensor] = preprocessor.preprocess(content, Modality.TEXT)
        results = [[(content[i], preprocessed_text_list[i])] for i in range(len(content))]
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