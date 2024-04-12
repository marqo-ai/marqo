from timeit import default_timer as timer
from typing import List, Optional, Union, Iterable, Sequence, Dict, Any, Tuple
import numpy as np

import marqo.core.unstructured_vespa_index.common as unstructured_common
from marqo import marqo_docs
from marqo import exceptions as base_exceptions
from marqo.api import exceptions as api_exceptions
from marqo.core import exceptions as core_exceptions
# We depend on _httprequests.py for now, but this may be replaced in the future, as
# _httprequests.py is designed for the client
from marqo.config import Config
from marqo.core import constants

from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models.marqo_index import IndexType
from marqo.core.models.marqo_index import MarqoIndex, FieldType, UnstructuredMarqoIndex, StructuredMarqoIndex
from marqo.core.models.marqo_query import MarqoTensorQuery, MarqoLexicalQuery
from marqo.core.structured_vespa_index.structured_vespa_index import StructuredVespaIndex
from marqo.core.unstructured_vespa_index import unstructured_validation as unstructured_index_add_doc_validation
from marqo.core.unstructured_vespa_index.unstructured_vespa_index import UnstructuredVespaIndex
from marqo.core.vespa_index import for_marqo_index as vespa_index_factory
from marqo.s2_inference import errors as s2_inference_errors
from marqo.s2_inference import s2_inference
from marqo.s2_inference.clip_utils import _is_image
from marqo.s2_inference.processing import image as image_processor
from marqo.s2_inference.processing import text as text_processor
from marqo.s2_inference.reranking import rerank
from marqo.tensor_search import delete_docs
from marqo.tensor_search import enums
from marqo.tensor_search import index_meta_cache
from marqo.tensor_search import utils, validation, add_docs, tensor_search
from marqo.tensor_search.enums import (
    Device, TensorField, SearchMethod, EnvVars
)
from marqo.tensor_search.index_meta_cache import get_cache, get_index
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from marqo.tensor_search.models.api_models import BulkSearchQueryEntity, ScoreModifier
from marqo.tensor_search.models.delete_docs_objects import MqDeleteDocsRequest
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.models.search import Qidx, JHash, SearchContext, VectorisedJobs, VectorisedJobPointer
from marqo.tensor_search.telemetry import RequestMetricsStore
from marqo.tensor_search.tensor_search_logging import get_logger
from marqo.vespa.exceptions import VespaStatusError
from marqo.vespa.models import VespaDocument, FeedBatchResponse, QueryResult

logger = get_logger(__name__)


def embed_content(
                config: Config, content: Union[Union[str, Dict[str, float]], List[Union[str, Dict[str, float]]]],
                index_name: str, device: str = None,
                image_download_headers: Optional[Dict] = None,
                model_auth: Optional[ModelAuth] = None
                ) -> List[List[float]]:
    """
    Use the index's model to embed the content

    Returns:
            List of embeddings corresponding to the content. If content is a list, the return list will be in the same order.
            If content is a string, the return list will only have 1 item.
    """

    # Content validation is done in API model layer
    t0 = timer()

    # Determine device
    if device is None:
        selected_device = utils.read_env_vars_and_defaults("MARQO_BEST_AVAILABLE_DEVICE")
        if selected_device is None:
            raise base_exceptions.InternalError("Best available device was not properly determined on Marqo startup.")
        logger.debug(f"No device given for search. Defaulting to best available device: {selected_device}")
    else:
        selected_device = device

    # Generate input for the vectorise pipeline (Preprocessing)
    RequestMetricsStore.for_request().start("embed.query_preprocessing")
    marqo_index = index_meta_cache.get_index(config=config, index_name=index_name)

    # Transform content to list if it is not already
    if isinstance(content, List):
        content_list = content
    elif isinstance(content, str) or isinstance(content, Dict):
        content_list = [content]
    else:
        raise base_exceptions.InternalError(f"Content type {type(content)} is not supported for embed endpoint.")

    queries = []
    for content_entry in content_list:
        queries.append(
            # TODO (future): Change to different object with only the necessary fields. Do the same with search.
            BulkSearchQueryEntity(
                q=content_entry,
                index=marqo_index,
                image_download_headers=image_download_headers,
                modelAuth=model_auth
                # TODO: Check if it's fine that we leave out the other parameters
            )
        )
    RequestMetricsStore.for_request().stop("embed.query_preprocessing")

    # Vectorise the queries
    with RequestMetricsStore.for_request().time(f"embed.vector_inference_full_pipeline"):
        qidx_to_vectors: Dict[Qidx, List[float]] = tensor_search.run_vectorise_pipeline(config, queries, selected_device)
    embeddings = list(qidx_to_vectors.values())

    # Record time and return final result
    time_taken = timer() - t0
    embeddings_final_result = {
        "content": content,
        "embeddings": embeddings,
        "processingTimeMs": round(time_taken * 1000)
    }
    logger.debug(f"embed request completed with total processing time: {(time_taken):.3f}s.")

    return embeddings_final_result




