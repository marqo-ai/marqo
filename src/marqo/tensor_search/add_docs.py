"""Functions used to fulfill the add_documents endpoint"""
import concurrent
import copy
import math
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import ContextManager
import threading
from queue import Queue
import torch

import logging
from typing import List, Dict

import numpy as np
import PIL
from PIL.ImageFile import ImageFile
from torchvision.transforms import Compose

import marqo.exceptions as base_exceptions
from marqo.core.models.marqo_index import *
from marqo.s2_inference import clip_utils
from marqo.s2_inference.s2_inference import is_preprocess_image_model, load_multimodal_model_and_get_preprocessors, \
    infer_modality, Modality
from marqo.s2_inference.errors import UnsupportedModalityError, S2InferenceError
from marqo.tensor_search.add_docs_utils import StreamingMediaProcessor
from marqo.tensor_search import enums
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.telemetry import RequestMetricsStore, RequestMetrics

from marqo.s2_inference.models.model_type import ModelType

logger = logging.getLogger(__name__)

class MemoryPool:
    def __init__(self, total_size: int, chunk_size: int):
        self.total_size = total_size
        self.chunk_size = chunk_size
        self.pool = Queue(maxsize=total_size // chunk_size)

        for _ in range(self.pool.maxsize):
            self.pool.put(bytearray(chunk_size))

    def get_chunk(self):
        return self.pool.get()

    def return_chunk(self, chunk):
        self.pool.put(chunk)


def threaded_download_and_preprocess_content(allocated_docs: List[dict], 
                                                content_repo: dict, 
                                                tensor_fields: List[str],
                                                image_download_headers: dict,
                                                device: str = None,
                                                memory_pool: Optional[MemoryPool] = None, # Optional for now
                                                download_headers: Optional[Dict] = None,  # Optional for now
                                                metric_obj: Optional[RequestMetrics] = None,
                                                preprocessors: Optional[Dict[str, Compose]] = None,
                                                marqo_index: Optional[MarqoIndex] = None) -> None:
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
    # Generate pseudo-unique ID for thread metrics.
    _id = f'image_download.{threading.get_ident()}'
    TIMEOUT_SECONDS = 3
    if metric_obj is None:  # Occurs predominately in testing.
        metric_obj = RequestMetricsStore.for_request()
        RequestMetricsStore.set_in_request(metrics=metric_obj)

    with metric_obj.time(f"{_id}.thread_time"):
        for doc in allocated_docs:
            for field in list(doc):
                if field not in tensor_fields:
                    continue
                if isinstance(doc[field], str):
                    modality = infer_modality(doc[field])
                    #print(f"from threaded_download_and_preprocess_content, modality: {modality}")
                    if modality == Modality.IMAGE: # or clip_utils._is_image(doc[field]):
                        #print(f"from threaded_download_and_preprocess_content, modality is IMAGE")
                        # Existing logic
                        if doc[field] in content_repo:
                            continue
                        try:
                            content_repo[doc[field]] = clip_utils.load_image_from_path(doc[field], image_download_headers,
                                                                                     timeout_ms=int(
                                                                                         TIMEOUT_SECONDS * 1000),
                                                                                     metrics_obj=metric_obj)
                        except PIL.UnidentifiedImageError as e:
                            content_repo[doc[field]] = e
                            metric_obj.increment_counter(f"{doc.get(field, '')}.UnidentifiedImageError")
                            continue
                        # preprocess image to tensor
                        if preprocessors is not None and preprocessors['image'] is not None:
                            ##print(f"preprocessors['image']: {preprocessors['image']}")
                            ##print(f"device: {device}")
                            if not device or not isinstance(device, str):
                                raise ValueError("Device must be provided for preprocessing images")
                            try:
                                #print(f"from threaded_download_and_preprocess_content, trying to preprocess image")
                                content_repo[doc[field]] = preprocessors['image'](content_repo[doc[field]]).to(device)
                                #print(f"content_repo[doc[field]]: {content_repo[doc[field]]}")
                            except OSError as e:
                                if "image file is truncated" in str(e):
                                    content_repo[doc[field]] = e
                                    metric_obj.increment_counter(f"{doc.get(field, '')}.OSError")
                                    continue
                                else:
                                    raise e

                    if modality in [Modality.VIDEO, Modality.AUDIO]:
                        if marqo_index.model.properties.get('type') not in [ModelType.LanguageBind]:
                            ##print(f"from threaded_download_and_preprocess_content, model is not a Multimodal model")
                            content_repo[doc[field]] = UnsupportedModalityError(f"Model {marqo_index.model.name} is not a Multimodal model and does not support {modality}")
                            continue
                        #print(f"from threaded_download_and_preprocess_content, modality is VIDEO or AUDIO")
                        try:
                            processed_chunks = download_and_chunk_media(doc[field], download_headers, modality, marqo_index, preprocessors)
                            content_repo[doc[field]] = processed_chunks
                        except Exception as e:
                            logger.error(f"Error processing {modality} file: {str(e)}")
                            content_repo[doc[field]] = S2InferenceError(f"Error processing {modality} file: {str(e)}")

                # For multimodal tensor combination
                elif isinstance(doc[field], dict):
                    for sub_field in list(doc[field].values()):
                        if isinstance(sub_field, str) and clip_utils._is_image(sub_field):
                            if sub_field in content_repo:
                                continue
                            try:
                                content_repo[sub_field] = clip_utils.load_image_from_path(
                                    sub_field,
                                    image_download_headers,
                                    timeout=TIMEOUT_SECONDS,
                                    metrics_obj=metric_obj
                                )
                            except PIL.UnidentifiedImageError as e:
                                content_repo[sub_field] = e
                                metric_obj.increment_counter(f"{doc.get(field, '')}.UnidentifiedImageError")
                                continue
    

def download_and_chunk_media(url: str, headers: dict, modality: Modality, marqo_index: MarqoIndex, preprocessors = dict) -> List[Dict[str, torch.Tensor]]:
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB in bytes

    processor = StreamingMediaProcessor(url, headers, modality, marqo_index, preprocessors)
    
    if processor.total_size > MAX_FILE_SIZE:
        raise ValueError(f"File size ({processor.total_size / 1024 / 1024:.2f} MB) exceeds the maximum allowed size of 100 MB")

    return processor.process_media()


@contextmanager
def download_and_preprocess_content(docs: List[dict], thread_count: int, tensor_fields: List[str],
                                    image_download_headers: dict,
                                    model_name: str,
                                    normalize_embeddings: bool,
                                    download_headers: Optional[Dict] = None,  # Optional for now
                                    model_properties: Optional[Dict] = None,
                                    model_auth: Optional[ModelAuth] = None,
                                    device: Optional[str] = None,
                                    patch_method_exists: bool = False,
                                    marqo_index: Optional[MarqoIndex] = None,
                                    ) -> ContextManager[dict]:
    
    #print(f"from download_and_preprocess_content, marqo_index: {marqo_index}")
    content_repo = {}  # for image/video/audio
    memory_pool = MemoryPool(total_size=500 * 1024 * 1024, chunk_size=20 * 1024 * 1024)  # 500 MB total, 20 MB chunks

    docs_per_thread = math.ceil(len(docs) / thread_count)
    copied = copy.deepcopy(docs)

    model, preprocessors = load_multimodal_model_and_get_preprocessors(
        model_name=model_name,
        model_properties=model_properties,
        device=device,
        model_auth=model_auth,
        normalize_embeddings=normalize_embeddings
    )
    #print(f"from download_and_preprocess_content, preprocessors: {preprocessors}")

    if not is_preprocess_image_model(model_properties) or patch_method_exists:
        #print(f"from download_and_preprocess_content, is_preprocess_image_model(model_properties): {is_preprocess_image_model(model_properties)}")
        #print(f"from download_and_preprocess_content, patch_method_exists: {patch_method_exists}")
        preprocessors['image'] = None

    try:
        m = [RequestMetrics() for i in range(thread_count)]
        thread_allocated_docs = [copied[i: i + docs_per_thread] for i in range(len(copied))[::docs_per_thread]]
        download_headers={}
        #print(f"from download_and_preprocess_content, thread_allocated_docs: {thread_allocated_docs}")
        with ThreadPoolExecutor(max_workers=len(thread_allocated_docs)) as executor:
            futures = [executor.submit(threaded_download_and_preprocess_content,
                           allocation, 
                           content_repo, 
                           tensor_fields,
                           image_download_headers, 
                           device,
                           memory_pool, 
                           download_headers, 
                           m[i], 
                           preprocessors,
                           marqo_index)
           for i, allocation in enumerate(thread_allocated_docs)]


            # Unhandled exceptions will be raised here.
            # We only raise the first exception if there are multiple exceptions
            for future in concurrent.futures.as_completed(futures):
                _ = future.result()

        # Fix up metric_obj to make it not mention thread-ids
        metric_obj = RequestMetricsStore.for_request()
        metric_obj = RequestMetrics.reduce_from_list([metric_obj] + m)
        metric_obj.times = reduce_thread_metrics(metric_obj.times)
        #print(f"from download_and_preprocess_content, content_repo: {content_repo}")
        yield content_repo
    finally:
        for p in content_repo.values():
            if isinstance(p, ImageFile):
                p.close()
            elif isinstance(p, (list, np.ndarray)):
                # Clean up video/audio chunks if necessary
                pass

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
            raise base_exceptions.InternalError(
                f"Invalid dict field {field_name}. Could not find field in mappings object.")

        if mappings[field_name]["type"] == enums.MappingsObjectType.multimodal_combination:
            return enums.MappingsObjectType.multimodal_combination
        elif mappings[field_name]["type"] == enums.MappingsObjectType.custom_vector:
            return enums.MappingsObjectType.custom_vector
        else:
            raise base_exceptions.InternalError(
                f"Invalid dict field type: '{mappings[field_name]['type']}' for field: '{field_name}' in mappings. Must be one of {[t.value for t in enums.MappingsObjectType]}")
    else:
        return None
