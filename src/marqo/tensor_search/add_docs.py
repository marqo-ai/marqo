"""Functions used to fulfill the add_documents endpoint"""
import copy
import functools
import math
import threading
import warnings
from typing import List
import PIL
from marqo.s2_inference.clip_utils import _is_image, load_image_from_path


def threaded_download_images(allocated_docs: List[dict], image_repo: dict) -> None:
    """A thread calls this function to download images for its allocated documents

    This should be called only if treat URLs as images is True
    """
    for doc in allocated_docs:
        for field in list(doc):
            if isinstance(doc[field], str) and _is_image(doc[field]):
                try:
                    image_repo[doc[field]] = load_image_from_path(doc[field])
                except PIL.UnidentifiedImageError:
                    image_repo[doc[field]] = None
                    continue


def download_images(docs: List[dict], thread_count: int) -> dict:
    """Concurrently downloads images from each doc, storing them into the image_repo dict

    This should be called only if treat URLs as images is True

    Returns:
         An image repo: a dict <image pointer>:<image data>
    """

    docs_per_thread = math.ceil(len(docs)/thread_count)
    copied = copy.deepcopy(docs)
    image_repo = dict()
    thread_allocated_docs = [copied[i: i + docs_per_thread] for i in range(len(copied))[::docs_per_thread]]
    threads = [threading.Thread(target=threaded_download_images, args=(allocation, image_repo))
               for allocation in thread_allocated_docs]
    for th in threads:
        th.start()

    for th in threads:
        th.join()
    return image_repo

