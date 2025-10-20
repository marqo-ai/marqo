"""Functions used to fulfill the add_documents endpoint"""

import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Union

import PIL
from PIL.Image import Image
from torch import Tensor

from inference_orchestrator.api.telemetry import RequestMetrics, RequestMetricsStore
from inference_orchestrator.schemas.api import (
    AudioPreprocessingConfig,
    ImagePreprocessingConfig,
    InferenceErrorModel,
    Modality,
    VideoPreprocessingConfig,
)
from inference_orchestrator.services.errors import (
    MediaDownloadError,
    PreprocessingError,
)
from inference_orchestrator.services.media_download_and_preprocess.image_download import (
    load_image_from_path,
)

logger = logging.getLogger(__name__)

PreprocessedContent = list[tuple[str, Union[str, any]]]


def threaded_download_and_preprocess_content(
    allocated_content: list[str],
    preprocessor,
    preprocessing_config: Union[
        ImagePreprocessingConfig, AudioPreprocessingConfig, VideoPreprocessingConfig
    ],
    metric_obj: Optional[RequestMetrics] = None,
    return_individual_error: bool = True,
) -> list[PreprocessedContent]:
    """
    A thread calls this function to download media(images, audio, video) for its allocated contents.

    Args:
        allocated_content: The content to be downloaded and preprocessed by this thread, normally a list of URLs.
        preprocessor: The preprocessor to be used for preprocessing the content. E.g., OpenCLIPPreprocessor
        preprocessing_config: The preprocessing configuration to be used for preprocessing the content.
        metric_obj: The telemetry object to be used for measuring the time taken for each thread.
        return_individual_error: Whether to return individual errors or raise them.
            If True, individual errors are returned as a InferenceErrorModel object in the results list,
            otherwise, they are raised.
    Returns:
        A list of preprocessed content.
    """
    modality = preprocessing_config.modality

    if modality == Modality.IMAGE:
        return _threaded_download_and_preprocess_image(
            allocated_content,
            preprocessor,
            preprocessing_config,
            metric_obj,
            return_individual_error,
        )
    else:
        raise ValueError(f"Unsupported modality: {modality}")


def _threaded_download_and_preprocess_image(
    allocated_content: list[str],
    preprocessor,
    preprocessing_config: ImagePreprocessingConfig,
    metric_obj: Optional[RequestMetrics] = None,
    return_individual_error: bool = True,
) -> list[PreprocessedContent]:
    """A thread calls this function to download images for its allocated contents.

    Args:
        allocated_content: A list of URLs to be downloaded and preprocessed by this thread.
        preprocessor: The preprocessor to be used for preprocessing the image. E.g., OpenCLIPPreprocessor
        preprocessing_config: The preprocessing configuration to be used for preprocessing the image. This includes
            the image download timeout and the image download headers.
        return_individual_error: If True, collect individual errors in the thread_results, otherwise, raise an error.
        metric_obj: The telemetry object to be used for measuring the time taken for each thread.

    Ret
    """
    _id = f"media_download.{preprocessing_config.modality}.{threading.get_ident()}"
    thread_results: list[Union[InferenceErrorModel, list[Tuple[str, Tensor]]]] = []
    with metric_obj.time(f"{_id}.thread_time"):
        for url in allocated_content:
            try:
                image = load_image_from_path(
                    url,
                    preprocessing_config.download_header,
                    timeout_ms=preprocessing_config.download_timeout_ms,
                    metrics_obj=metric_obj,
                )
            except PIL.UnidentifiedImageError as e:
                metric_obj.increment_counter(f"{url}.UnidentifiedImageError")
                if return_individual_error:
                    thread_results.append(InferenceErrorModel(error_message=str(e)))
                else:
                    raise MediaDownloadError(str(e))
                continue
            if isinstance(image, Image):
                try:
                    preprocessed_image: List[Tensor] = preprocessor.preprocess(
                        [image], preprocessing_config.modality
                    )
                except OSError as e:
                    if "image file is truncated" in str(e):
                        if return_individual_error:
                            thread_results.append(
                                InferenceErrorModel(
                                    error_message=f"Image file is truncated: {url}"
                                )
                            )
                        else:
                            raise PreprocessingError(f"Image file is truncated: {url}")
                        continue
                    else:
                        raise e
                thread_results.append([(url, preprocessed_image[0])])
            else:
                if return_individual_error:
                    thread_results.append(
                        InferenceErrorModel(
                            error_message=f"Unexpected image type: {type(image)} "
                            f"for image: {url}"
                        )
                    )
                else:
                    raise ValueError(
                        f"Unexpected image type: {type(image)} for image: {url}"
                    )
    return thread_results


def process_batch(
    content: list[str],
    preprocessor,
    preprocessing_config: Union[
        ImagePreprocessingConfig, AudioPreprocessingConfig, VideoPreprocessingConfig
    ],
    return_individual_error: bool = True,
) -> list[PreprocessedContent]:
    results: list[PreprocessedContent] = []

    thread_count = preprocessing_config.download_thread_count

    content_per_thread = math.ceil(len(content) / thread_count)
    m = [RequestMetrics() for _ in range(thread_count)]
    thread_allocated_docs = [
        content[i : i + content_per_thread]
        for i in range(0, len(content), content_per_thread)
    ]

    # Using the map function to ensure the results are in the same order as the input
    with ThreadPoolExecutor(max_workers=len(thread_allocated_docs)) as executor:
        results_nested = list(
            executor.map(
                lambda args: threaded_download_and_preprocess_content(*args),
                [
                    (
                        allocation,
                        preprocessor,
                        preprocessing_config,
                        m[i],
                        return_individual_error,
                    )
                    for i, allocation in enumerate(thread_allocated_docs)
                ],
            )
        )

    for partial_result in results_nested:
        results.extend(partial_result)

    # Fix up metric_obj to make it not mention thread-ids
    metric_obj = RequestMetricsStore.for_request()
    metric_obj = RequestMetrics.reduce_from_list([metric_obj] + m)
    metric_obj.times = reduce_thread_metrics(metric_obj.times)
    return results


def reduce_thread_metrics(data):
    """Reduce the metrics from each thread, as if they were run in a single thread.

    e.g.
    ```
    {
        "media_download.image.700.thread_time": 1373.271582997404,
        "media_download.image.700.https://www.ai-nc.com/images/pages/heat-map.png": 52.985392,
        "media_download.image.729.thread_time": 53.297404,
        "media_download.image.729.https://www.ai-nc.com/images/pages/heat-map.png": 2052.617332985392,
    }
    ```
    Becomes
    ```
    {
        "media_download.image.thread_time": [1373.271582997404, 53.297404],
        "media_download.image.https://www.ai-nc.com/images/pages/heat-map.png": [2052.617332985392, 52.985392],
    }
    ```
    Only applies to times that start with `media_download`.
    """
    result = {}
    for key, value in data.items():
        if key.startswith("media_download."):
            parts = key.split(".")
            if len(parts) < 4:
                continue
            new_key = ".".join(parts[0:2] + parts[3:])
            if new_key in result:
                if isinstance(result[new_key], list):
                    result[new_key].append(value)
                else:
                    result[new_key] = [result[new_key], value]
            else:
                result[new_key] = value
    return result
