"""Functions used to fulfill the add_documents endpoint"""
import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor

import PIL
from PIL.Image import Image

from marqo.inference.media_download_and_preprocess.image_download import load_image_from_path
from marqo.inference.type import *
from marqo.tensor_search import utils
from marqo.tensor_search.enums import EnvVars
from marqo.tensor_search.telemetry import RequestMetricsStore, RequestMetrics

logger = logging.getLogger(__name__)


def threaded_download_and_preprocess_content(
        allocated_content: list[str],
        preprocessor,
        modality: Modality,
        media_download_headers: Optional[Dict] = None,
        download_timeout_ms: int = 3000,
        audio_video_preprocessing_config: Union[None, VideoPreprocessingConfig, AudioPreprocessingConfig] = None,
        metric_obj: Optional[RequestMetrics] = None,
        return_individual_error: bool = True,
) -> list[PreprocessedContent]:
    """A thread calls this function to download images for its allocated documents

    This should be called only if treat URLs as images is True.

    Args:
        allocated_docs: docs with images to be downloaded by this thread,
        media_repo: dictionary that will be mutated by this thread. It will add media
            as values and the URLs as keys
        tensor_fields: A tuple of tensor_fields. Images will be downloaded for these fields only.
        media_download_headers: A dict of headers for image download. Can be used
            to authenticate image downloads
        return_individual_error: If True, collect individual errors in the thread_results, otherwise, raise an error.
    Side Effects:
        Adds members to the image_repo dict. Each key is a string which is identified as a URL.
        Each value is either a PIL image, or UnidentifiedImageError, if there were any errors encountered retrieving
        the image.
        For example:
        {
            'https://google.com/my_dog.png': InferenceErrorModel, # error because such an image doesn't exist
            'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png': <PIL image>
        }
    Returns:
        None

    """
    _id = f'image_download.{threading.get_ident()}'
    thread_results: list[Union[InferenceErrorModel, list[Tuple[str, Tensor]]]] = []
    with metric_obj.time(f"{_id}.thread_time"):
        for url in allocated_content:
            try:
                image = load_image_from_path(
                    url, media_download_headers, timeout_ms=download_timeout_ms, metrics_obj=metric_obj
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
                    preprocessed_image: List[Tensor] = preprocessor.preprocess([image], modality)
                except OSError as e:
                    if "image file is truncated" in str(e):
                        if return_individual_error:
                            thread_results.append(InferenceErrorModel(error_message=f"Image file is truncated: {url}"))
                        else:
                            raise PreprocessingError(f"Image file is truncated: {url}")
                        continue
                    else:
                        raise e
                thread_results.append([(url, preprocessed_image[0])])
            else:
                if return_individual_error:
                    thread_results.append(InferenceErrorModel(error_message=f"Unexpected image type: {type(image)} "
                                                                            f"for image: {url}"))
                else:
                    raise ValueError(f"Unexpected image type: {type(image)} for image: {url}")
    return thread_results



def _enable_video_gpu_acceleration() -> bool:
    """A helper function to determine if the video decoding should be done on the GPU.

    The environment variable MARQO_ENABLE_VIDEO_GPU_ACCELERATION is set on marqo start_on script.
    """
    return utils.read_env_vars_and_defaults(EnvVars.MARQO_ENABLE_VIDEO_GPU_ACCELERATION) == 'TRUE'


def process_batch(
        content: list[str],
        preprocessor,
        modality: Modality,
        thread_count: int,
        media_download_headers: Optional[Dict] = None,
        download_timeout_ms: int = 3000,
        audio_video_preprocessing_config: Union[None, AudioPreprocessingConfig, VideoPreprocessingConfig] = None,
        return_individual_error: bool = True
) -> list[PreprocessedContent]:

    results: list[PreprocessedContent] = []

    content_per_thread = math.ceil(len(content) / thread_count)
    m = [RequestMetrics() for _ in range(thread_count)]
    thread_allocated_docs = [content[i: i + content_per_thread] for i in range(0, len(content), content_per_thread)]

    # Using the map function to ensure the results are in the same order as the input
    with ThreadPoolExecutor(max_workers=len(thread_allocated_docs)) as executor:
        results_nested = list(executor.map(
            lambda args: threaded_download_and_preprocess_content(*args),
            [
                (
                    allocation,
                    preprocessor,
                    modality,
                    media_download_headers,
                    download_timeout_ms,
                    audio_video_preprocessing_config,
                    m[i],
                    return_individual_error
                )
                for i, allocation in enumerate(thread_allocated_docs)
            ]
        ))

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
        "image_download.700.thread_time": 1373.271582997404,
        "image_download.700.https://www.ai-nc.com/images/pages/heat-map.png": 52.985392,
        "image_download.729.thread_time": 53.297404,
        "image_download.729.https://www.ai-nc.com/images/pages/heat-map.png": 2052.617332985392,
    }
    ```
    Becomes
    ```
    {
        "image_download.thread_time": [1373.271582997404, 53.297404],
        "image_download.https://www.ai-nc.com/images/pages/heat-map.png": [2052.617332985392, 52.985392],
    }
    ```
    Only applies to times that start with `image_download`.
    """
    result = {}
    for key, value in data.items():
        if key.startswith("image_download."):
            parts = key.split('.')
            new_key = '.'.join(parts[0:1] + parts[2:]) if parts[1] != 'full_time' else key
            if new_key in result:
                if isinstance(result[new_key], list):
                    result[new_key].append(value)
                else:
                    result[new_key] = [result[new_key], value]
            else:
                result[new_key] = value
    return result