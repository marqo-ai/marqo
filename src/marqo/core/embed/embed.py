from timeit import default_timer as timer
from typing import List, Optional, Union, Iterable, Sequence, Dict, Any, Tuple
import numpy as np

import pydantic
import marqo.core.unstructured_vespa_index.common as unstructured_common
from marqo import marqo_docs
from marqo import exceptions as base_exceptions
from marqo.api import exceptions as api_exceptions
from marqo.core import exceptions as core_exceptions
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
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from marqo.tensor_search.models.api_models import BulkSearchQueryEntity, ScoreModifier
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.models.search import Qidx, JHash, SearchContext, VectorisedJobs, VectorisedJobPointer
from marqo.tensor_search.telemetry import RequestMetricsStore
from marqo.tensor_search.tensor_search_logging import get_logger
from marqo.vespa.exceptions import VespaStatusError
from marqo.vespa.models import VespaDocument, FeedBatchResponse, QueryResult
from marqo.vespa.vespa_client import VespaClient

logger = get_logger(__name__)


class Embed:
    def __init__(self, vespa_client: VespaClient, index_management: IndexManagement, default_device: str):
        self.vespa_client = vespa_client
        self.index_management = index_management
        self.default_device = default_device

    @pydantic.validator('default_device')
    def validate_default_device(cls, value):
        if not value:
            raise ValueError("Default Device cannot be 'None'. Marqo default device must have been declared upon startup.")
        return value

    def embed_content(
                    self, content: Union[str, Dict[str, float], List[Union[str, Dict[str, float]]]],
                    index_name: str, device: str = None, image_download_headers: Optional[Dict] = None,
                    model_auth: Optional[ModelAuth] = None
                    ) -> List[List[float]]:
        """
        Use the index's model to embed the content

        Returns:
                List of embeddings corresponding to the content. If content is a list, the return list will be in the same order.
                If content is a string, the return list will only have 1 item.
        """

        # TODO: Remove this config constructor once vectorise pipeline doesn't need it. Just pass the vespa client
        # and index management objects.
        from marqo import config
        from marqo.tensor_search import utils, validation, tensor_search, index_meta_cache
        temp_config = config.Config(
            vespa_client=self.vespa_client,
            index_management=self.index_management,
            default_device=self.default_device
        )

        # Set default device if not provided
        if device is None:
            device = self.default_device

        # Content validation is done in API model layer
        t0 = timer()

        # Generate input for the vectorise pipeline (Preprocessing)
        RequestMetricsStore.for_request().start("embed.query_preprocessing")
        marqo_index = index_meta_cache.get_index(config=temp_config, index_name=index_name)

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
            qidx_to_vectors: Dict[Qidx, List[float]] = tensor_search.run_vectorise_pipeline(temp_config, queries, device)
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




