"""Functions used to fulfill the add_documents endpoint"""
import copy
import functools
import math
import threading
import warnings
from typing import List, Tuple
import PIL
from marqo.s2_inference import clip_utils


def threaded_download_images(allocated_docs: List[dict], image_repo: dict,
                             non_tensor_fields: Tuple, image_download_headers: dict) -> None:
    """A thread calls this function to download images for its allocated documents

    This should be called only if treat URLs as images is True.

    Args:
        allocated_docs: docs with images to be downloaded by this thread,
        image_repo: dictionary that will be mutated by this thread. It will add PIL images
            as values and the URLs as keys
        non_tensor_fields: A tuple of non_tensor_fields. No images will be downloaded for
            these fields
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
    TIMEOUT_SECONDS=3
    for doc in allocated_docs:
        for field in list(doc):
            if field in non_tensor_fields:
                continue
            if isinstance(doc[field], str) and clip_utils._is_image(doc[field]):
                if doc[field] in image_repo:
                    continue
                try:
                    image_repo[doc[field]] = clip_utils.load_image_from_path(doc[field], image_download_headers, timeout=TIMEOUT_SECONDS)
                except PIL.UnidentifiedImageError as e:
                    image_repo[doc[field]] = e
                    continue
            # For multimodal tensor combination
            elif isinstance(doc[field], dict):
                for sub_field in list(doc[field].values()):
                    if isinstance(sub_field, str) and clip_utils._is_image(sub_field):
                        if sub_field in image_repo:
                            continue
                        try:
                            image_repo[sub_field] = clip_utils.load_image_from_path(sub_field, image_download_headers,
                                                                          timeout=TIMEOUT_SECONDS)
                        except PIL.UnidentifiedImageError as e:
                            image_repo[sub_field] = e
                            continue


def download_images(docs: List[dict], thread_count: int, non_tensor_fields: Tuple, image_download_headers: dict) -> dict:
    """Concurrently downloads images from each doc, storing them into the image_repo dict
    Args:
        docs: docs with images to be downloaded. These will be allocated to each thread
        thread_count: number of threads to spin up
        non_tensor_fields: A tuple of non_tensor_fields. No images will be downloaded for
            these fields
        image_download_headers: A dict of image download headers for authentication.
    This should be called only if treat URLs as images is True

    Returns:
         An image repo: a dict <image pointer>:<image data>
    """

    docs_per_thread = math.ceil(len(docs)/thread_count)
    copied = copy.deepcopy(docs)
    image_repo = dict()
    thread_allocated_docs = [copied[i: i + docs_per_thread] for i in range(len(copied))[::docs_per_thread]]
    threads = [threading.Thread(target=threaded_download_images, args=(allocation, image_repo, non_tensor_fields, image_download_headers))
               for allocation in thread_allocated_docs]
    for th in threads:
        th.start()

    for th in threads:
        th.join()
    return image_repo

