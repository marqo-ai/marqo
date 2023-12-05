"""Functions used to fulfill the add_documents endpoint"""
import copy
from contextlib import contextmanager

import math
import threading
import random
import uuid
from typing import List, Optional, ContextManager, Dict
import PIL
import numpy as np
from PIL.ImageFile import ImageFile
from marqo.s2_inference import clip_utils, clap_utils
from marqo.tensor_search.telemetry import RequestMetricsStore, RequestMetrics
import marqo.errors as errors
from marqo.tensor_search import utils
from marqo.tensor_search import enums
from marqo.tensor_search import constants
from marqo.tensor_search.models.index_info import IndexInfo

def load_media_from_path(
        media_repo: dict,
        media_path: str,
        download_headers: dict,
        media_type: enums.MediaType,
        timeout: int = 15,
        metrics_obj: Optional[RequestMetrics] = None,
    ) -> None:
    """_summary_

    Args:
        media_repo (dict): The media repo to populate with data
        media_path (str): The path to the media
        download_headers (dict): Download headers for downloading media
        media_type (enums.MediaType): Enum describing the type of media
        timeout (int, optional): Seconds before timeout. Defaults to 15.
        metrics_obj (Optional[RequestMetrics], optional): Request detail tracking object. Defaults to None.

    Side Effects:
        Adds members to the media_repo dict. Each key is a string which is identified as a URL.
        
        Adds metrics to the metrics_obj.
    Raises:
        errors.InternalError: Error if an invalid media type enumeration is provided
    """

    # match media_type:
    #     case enums.MediaType.image:
    #         if not clip_utils._is_image(media_path):
    #             raise errors.InternalError(f"Invalid image path {media_path}")
    #         media_repo[media_path] = clip_utils.load_image_from_path(
    #             media_path, 
    #             download_headers, 
    #             timeout=timeout, 
    #             metrics_obj=metrics_obj
    #         )
    #     case enums.MediaType.audio:
    #         if not clap_utils._is_audio(media_path):
    #             raise errors.InternalError(f"Invalid audio path {media_path}")
    #         media_repo[media_path] = clap_utils.load_audio_from_path(
    #             media_path, 
    #             download_headers, 
    #             timeout=timeout, 
    #             metrics_obj=metrics_obj
    #         )
    #     case _:
    #         raise errors.InternalError(f"Invalid media type {media_type}")

    if media_type == enums.MediaType.image:
        if not clip_utils._is_image(media_path):
            raise errors.InternalError(f"Invalid image path {media_path}")
        media_repo[media_path] = clip_utils.load_image_from_path(
            media_path, 
            download_headers, 
            timeout=timeout, 
            metrics_obj=metrics_obj
        )
    elif media_type == enums.MediaType.audio:
        if not clap_utils._is_audio(media_path):
            raise errors.InternalError(f"Invalid audio path {media_path}")
        media_repo[media_path] = clap_utils.load_audio_from_path(
            media_path, 
            download_headers, 
            timeout=timeout, 
            metrics_obj=metrics_obj
        )
    else:
        raise errors.InternalError(f"Invalid media type {media_type}")


def threaded_download_media(
        allocated_docs: List[dict], 
        media_repo: dict, 
        media_type: enums.MediaType,
        tensor_fields: Optional[List[str]],
        non_tensor_fields: Optional[List[str]], 
        download_headers: Optional[dict],
        metric_obj: Optional[RequestMetrics] = None,
        timeout_seconds: int = 3,
    ) -> None:
    """A thread calls this function to download images for its allocated documents

    This should be called only if treat URLs as images is True.

    Args:
        allocated_docs: docs with images to be downloaded by this thread,
        media_repo: dictionary that will be mutated by this thread. It will add media
            as values and the URLs as keys
        media_type: The type of the media to be downloaded
        tensor_fields: A tuple of tensor_fields. Images will be downloaded for these fields only. Cannot be provided
            at the same time as `non_tensor_fields`.
        non_tensor_fields: A tuple of non_tensor_fields. No images will be downloaded for
            these fields. Cannot be provided at the same time as `tensor_fields`.
        download_headers: A dict of headers for image download. Can be used
            to authenticate image downloads
    Side Effects:
        Adds members to the image_repo dict. Each key is a string which is identified as a URL.
        Each value is either a PIL image, or UnidentifiedImageError, if there were any errors encountered retrieving
        the image.
        For example:
        {
            'https://google.com/my_dog.png': UnidentifiedImageError, # error because such an image doesn't exist
            'https://marqo-assets.s3.amazonaws.com/tests/images/ai_hippo_realistic.png': <PIL image>
        }
    Returns:
        None
    Raises:
        - InternalError if both or neither of tensor_fields and non_tensor_fields are provided. This validation should
        take place at API level and such invalid arguments are not expected to reach this function.

    """

    if tensor_fields is None == non_tensor_fields is None:
        raise errors.InternalError("Must provide exactly one of tensor_fields or non_tensor_fields")

    # Generate pseudo-unique ID for thread metrics.
    _id = uuid.uuid4().hex
    _id = f"image_download.{_id}"

    if metric_obj is None: # Occurs predominately in testing.
        metric_obj = RequestMetricsStore.for_request()
        RequestMetricsStore.set_in_request(metrics=metric_obj)

    with metric_obj.time(f"{_id}.thread_time"):
        for doc in allocated_docs:
            for field in list(doc):
                if not utils.is_tensor_field(field, tensor_fields, non_tensor_fields):
                    continue

                media_pointers: Dict[str, str] = {}

                if isinstance(doc[field], str):
                    media_pointers[field] = doc[field]
                elif isinstance(doc[field], dict):
                    mapping_field = doc[field]
                    for sub_field in mapping_field:
                        if isinstance(mapping_field[sub_field], str):
                            media_pointers[mapping_field] = sub_field
                        else:
                            continue

                for field in media_pointers:
                    if doc[field] in media_repo:
                        continue
                    try:
                        load_media_from_path(
                            media_repo, 
                            doc[field], 
                            download_headers, 
                            media_type, 
                            timeout=timeout_seconds, 
                            metrics_obj=metric_obj
                        ) # media_repo is passed by reference
                    except PIL.UnidentifiedImageError as e:
                        media_repo[doc[field]] = e
                        metric_obj.increment_counter(f"{doc.get(field, '')}.UnidentifiedImageError")
                        continue


@contextmanager
def download_media(
        docs: List[dict], 
        thread_count: int, 
        media_type: enums.MediaType,
        tensor_fields: Optional[List[str]],
        non_tensor_fields: Optional[List[str]], 
        download_headers: Optional[dict],
        timeout_seconds: int = 3,
    ) -> ContextManager[dict]:
    """Concurrently downloads images from each doc, storing them into the image_repo dict
    Args:
        docs: docs with images to be downloaded. These will be allocated to each thread
        thread_count: number of threads to spin up
        media_type: The type of media to download
        tensor_fields: A tuple of tensor_fields. Images will be downloaded for these fields only. Cannot be provided
            at the same time as `non_tensor_fields`.
        non_tensor_fields: A tuple of non_tensor_fields. No images will be downloaded for
            these fields. Cannot be provided at the same time as `tensor_fields`.
        download_headers: A dict of image download headers for authentication.
    This should be called only if treat URLs as images is True

    Returns:
         An image repo: a dict <image pointer>:<image data>

    Raises:
        - InternalError if both or neither of tensor_fields and non_tensor_fields are provided. This validation should
        take place at API level and such invalid arguments are not expected to reach this function.
    """

    if tensor_fields is None == non_tensor_fields is None:
        raise errors.InternalError("Must provide exactly one of tensor_fields or non_tensor_fields")

    docs_per_thread = math.ceil(len(docs)/thread_count)
    copied = copy.deepcopy(docs)
    media_repo = dict()

    try:
        m = [RequestMetrics() for i in range(thread_count)]
        thread_allocated_docs = [
            copied[i: i + docs_per_thread] for i in range(len(copied))[::docs_per_thread]
        ]
        threads = [
            threading.Thread(
                target=threaded_download_media, 
                args=(
                    allocation, 
                    media_repo,
                    media_type,
                    tensor_fields, 
                    non_tensor_fields, 
                    download_headers, 
                    m[i],
                    timeout_seconds
                )
            ) for i, allocation in enumerate(thread_allocated_docs)
        ]

        [th.start()for th in threads]

        [th.join()for th in threads]

        # Fix up metric_obj to make it not mention thread-ids
        metric_obj = RequestMetricsStore.for_request()
        metric_obj = RequestMetrics.reduce_from_list([metric_obj] + m)
        metric_obj.times = reduce_thread_metrics(metric_obj.times)
        yield media_repo
    finally:
        for p in media_repo.values():
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


def create_chunk_metadata(raw_document: dict) -> dict:
    """
    Creates a chunk metadata dictionary for a given document.
    This metadata will be put in each OpenSearch child document (chunk) to be used for filtering.

    We will only add values which are string, boolean, int, float, list or dictionary.
    """

    metadata = {}
    metadata_field_types = {str, bool, int, float, list, dict}
    for key, value in raw_document.items():
        for cls in metadata_field_types:
            if isinstance(value, cls):
                metadata[key] = value
                break
    return metadata


def determine_document_field_type(field_name: str, field_content, mappings: dict) -> enums.DocumentFieldType:
    """
    Determines the type of a document field
    using its name, content, and the add docs mappings object.

    3 Options:
    1. standard (str, int, float, bool, list)
    2. multimodal_combination (dict)
    3. custom_vector (dict)
    """

    if isinstance(field_content, dict):
        if field_name not in mappings:
            raise errors.InternalError(f"Invalid dict field {field_name}. Could not find field in mappings object.")
        
        if mappings[field_name]["type"] == enums.MappingsObjectType.multimodal_combination:
            return enums.DocumentFieldType.multimodal_combination
        elif mappings[field_name]["type"] == enums.MappingsObjectType.custom_vector:
            return enums.DocumentFieldType.custom_vector
        else:
            raise errors.InternalError(f"Invalid dict field type {field_name} in mappings. Must be one of {[t.value for t in enums.MappingsObjectType]}")
    
    return enums.DocumentFieldType.standard


def determine_text_chunk_prefix(request_level_prefix: str, index_info: IndexInfo) -> str:
    """
    Determines the text chunk prefix to be used for chunking text fields.
    This prefix will be added before each text chunk to be used for better inference.

    Logic:
    1. Prioritize request-level prefix
    2. If not provided, use override in text_preprocessing
    3. If not provided, use model_properties defined prefix
    4. If not provided, keep as None (will be handled by dict .get() method)
    """

    if request_level_prefix is not None:
        return request_level_prefix
    
    # Use override in text_preprocessing (if not None)
    index_settings = index_info.get_index_settings()
    if enums.IndexSettingsField.text_preprocessing in index_settings[enums.IndexSettingsField.index_defaults]:
        text_preproc = index_settings[enums.IndexSettingsField.index_defaults][enums.IndexSettingsField.text_preprocessing]
        if enums.IndexSettingsField.override_text_chunk_prefix in text_preproc:
            if text_preproc[enums.IndexSettingsField.override_text_chunk_prefix] is not None:
                return text_preproc[enums.IndexSettingsField.override_text_chunk_prefix]

    # Use model-defined prefix (None if it does not exist)
    model_prefix = index_info.get_model_properties().get(enums.ModelProperties.text_chunk_prefix)
    return model_prefix