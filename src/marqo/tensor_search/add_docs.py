"""Functions used to fulfill the add_documents endpoint"""
import copy
import math
import random
import threading
from contextlib import contextmanager
from typing import List, Optional, Tuple, ContextManager, Union

import PIL
from PIL.ImageFile import ImageFile

from marqo.s2_inference import clip_utils
from marqo.tensor_search.telemetry import RequestMetricsStore, RequestMetrics
from marqo.tensor_search import enums
from marqo.tensor_search import constants
import marqo.core.exceptions as core_exceptions
import marqo.exceptions as base_exceptions
from marqo.core.models.marqo_index import *


def threaded_download_images(allocated_docs: List[dict], image_repo: dict, tensor_fields: List[str],
                             image_download_headers: dict,
                             metric_obj: Optional[RequestMetrics] = None) -> None:
    """A thread calls this function to download images for its allocated documents

    This should be called only if treat URLs as images is True.

    Args:
        allocated_docs: docs with images to be downloaded by this thread,
        image_repo: dictionary that will be mutated by this thread. It will add PIL images
            as values and the URLs as keys
        tensor_fields: A tuple of tensor_fields. Images will be downloaded for these fields only.
        image_download_headers: A dict of headers for image download. Can be used
            to authenticate image downloads
    Side Effects:
        Adds members to the image_repo dict. Each key is a string which is identified as a URL.
        Each value is either a PIL image, or UnidentifiedImageError, if there were any errors encountered retrieving
        the image.
        For example:
        {
            'https://google.com/my_dog.png': UnidentifiedImageError, # error because such an image doesn't exist
            'https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png': <PIL image>
        }
    Returns:
        None

    """
    # TODO - We may not be handling errors in threads properly. Test introducing errors (e.g., call a method
    #  that doesn't exist) in this code and verify
    # Generate pseudo-unique ID for thread metrics.
    _id = hash("".join([d.get("_id", str(random.getrandbits(64))) for d in allocated_docs])) % 1000
    _id = f"image_download.{_id}"
    TIMEOUT_SECONDS = 3
    if metric_obj is None:  # Occurs predominately in testing.
        metric_obj = RequestMetricsStore.for_request()
        RequestMetricsStore.set_in_request(metrics=metric_obj)

    with metric_obj.time(f"{_id}.thread_time"):
        for doc in allocated_docs:
            for field in list(doc):
                if field not in tensor_fields:
                    continue
                if isinstance(doc[field], str) and clip_utils._is_image(doc[field]):
                    if doc[field] in image_repo:
                        continue
                    try:
                        image_repo[doc[field]] = clip_utils.load_image_from_path(doc[field], image_download_headers,
                                                                                 timeout=TIMEOUT_SECONDS,
                                                                                 metrics_obj=metric_obj)
                    except PIL.UnidentifiedImageError as e:
                        image_repo[doc[field]] = e
                        metric_obj.increment_counter(f"{doc.get(field, '')}.UnidentifiedImageError")
                        continue
                # For multimodal tensor combination
                elif isinstance(doc[field], dict):
                    for sub_field in list(doc[field].values()):
                        if isinstance(sub_field, str) and clip_utils._is_image(sub_field):
                            if sub_field in image_repo:
                                continue
                            try:
                                image_repo[sub_field] = clip_utils.load_image_from_path(
                                    sub_field,
                                    image_download_headers,
                                    timeout=TIMEOUT_SECONDS,
                                    metrics_obj=metric_obj
                                )
                            except PIL.UnidentifiedImageError as e:
                                image_repo[sub_field] = e
                                metric_obj.increment_counter(f"{doc.get(field, '')}.UnidentifiedImageError")
                                continue


@contextmanager
def download_images(docs: List[dict], thread_count: int, tensor_fields: List[str],
                    image_download_headers: dict) -> ContextManager[dict]:
    """Concurrently downloads images from each doc, storing them into the image_repo dict
    Args:
        docs: docs with images to be downloaded. These will be allocated to each thread
        thread_count: number of threads to spin up
        tensor_fields: A tuple of tensor_fields. Images will be downloaded for these fields only.
        image_download_headers: A dict of image download headers for authentication.
    This should be called only if treat URLs as images is True

    Returns:
         An image repo: a dict <image pointer>:<image data>
    """
    docs_per_thread = math.ceil(len(docs) / thread_count)
    copied = copy.deepcopy(docs)
    image_repo = dict()

    try:
        m = [RequestMetrics() for i in range(thread_count)]
        thread_allocated_docs = [copied[i: i + docs_per_thread] for i in range(len(copied))[::docs_per_thread]]
        threads = [threading.Thread(target=threaded_download_images, args=(allocation, image_repo,
                                                                           tensor_fields,
                                                                           image_download_headers, m[i]))
                   for i, allocation in enumerate(thread_allocated_docs)]

        for th in threads:
            th.start()

        for th in threads:
            th.join()

        # Fix up metric_obj to make it not mention thread-ids
        metric_obj = RequestMetricsStore.for_request()
        metric_obj = RequestMetrics.reduce_from_list([metric_obj] + m)
        metric_obj.times = reduce_thread_metrics(metric_obj.times)
        yield image_repo
    finally:
        for p in image_repo.values():
            if isinstance(p, ImageFile):
                p.close()


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


def determine_document_dict_field_type(field_name: str, field_content, mappings: dict) -> FieldType:
    """
    Only used for unstructured. Structured indexes have field types declared upon index creation.
    Determines the type of a document field if it is a dict
    using its name, content, and the add docs mappings object.
    3 Options:
    1. `None` if standard (str, int, float, bool, list)
    2. `MultimodalCombination` (dict)
    3. `CustomVector` (dict)
    4. Add other dict types as needed
    """

    if isinstance(field_content, dict):
        if field_name not in mappings:
            raise base_exceptions.InternalError(f"Invalid dict field {field_name}. Could not find field in mappings object.")

        if mappings[field_name]["type"] == enums.MappingsObjectType.multimodal_combination:
            return FieldType.MultimodalCombination
        elif mappings[field_name]["type"] == enums.MappingsObjectType.custom_vector:
            return FieldType.CustomVector
        else:
            raise base_exceptions.InternalError(f"Invalid dict field type: {mappings[field_name]['type']} for field: {field_name} in mappings. Must be one of {[t.value for t in enums.MappingsObjectType]}")
    else:
        return None