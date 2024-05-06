from timeit import default_timer as timer
from typing import List, Optional, Union, Dict
from enum import Enum

import pydantic

from marqo import exceptions as base_exceptions
from marqo.core.index_management.index_management import IndexManagement
from marqo.tensor_search.models.api_models import BulkSearchQueryEntity
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.models.search import Qidx
from marqo.tensor_search.telemetry import RequestMetricsStore
from marqo.tensor_search.tensor_search_logging import get_logger
from marqo.core.utils.prefix import determine_text_prefix
from marqo.vespa.vespa_client import VespaClient

logger = get_logger(__name__)

class EmbedContentType(Enum):
    Query = "query"
    Document = "document"

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
                    model_auth: Optional[ModelAuth] = None,
                    content_type: Optional[EmbedContentType] = EmbedContentType.Query
                    ) -> Dict:
        """
        Use the index's model to embed the content

        Returns:
                List of embeddings corresponding to the content. If content is a list, the return list will be in the same order.
                If content is a string, the return list will only have 1 item.
        """
        """
        NOTE: PARAMETER: content_type
        3 Options: ‘query’, ‘document’, None. Defaults to ‘query’.
        1. If the user wants to use the default text_query_prefix, leave it as ‘query’.
        2. If the user wants to use the default text_chunk_prefix, leave it as ‘document’.
        3. If the user wants a custom prefix, they must put it in the content itself.
        """


        # TODO: Remove this config constructor once vectorise pipeline doesn't need it. Just pass the vespa client
        # and index management objects.
        from marqo import config
        from marqo.tensor_search import tensor_search, index_meta_cache
        temp_config = config.Config(
            vespa_client=self.vespa_client,
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

        # Decide on the prefix 
        if content_type == EmbedContentType.Query:
            prefix = determine_text_prefix(None, marqo_index, "text_query_prefix")
        elif content_type == EmbedContentType.Document:
            prefix = determine_text_prefix(None, marqo_index, "text_chunk_prefix")
        elif content_type is None:
            prefix = ""
        else:
            raise ValueError(f"Invalid content_type: {content_type}. Must be EmbedContentType.Query, EmbedContentType.Document, or None.")
        
        queries = []
        for content_entry in content_list:
            queries.append(
                # TODO (future): Change to different object with only the necessary fields. Do the same with search.
                BulkSearchQueryEntity(
                    q=content_entry,
                    index=marqo_index,
                    image_download_headers=image_download_headers,
                    modelAuth=model_auth,
                    text_query_prefix=prefix
                    # TODO: Check if it's fine that we leave out the other parameters
                )
            )
        RequestMetricsStore.for_request().stop("embed.query_preprocessing")

        # Vectorise the queries
        with RequestMetricsStore.for_request().time(f"embed.vector_inference_full_pipeline"):
            qidx_to_vectors: Dict[Qidx, List[float]] = tensor_search.run_vectorise_pipeline(temp_config, queries, device)

        embeddings: List[List[float]] = list(qidx_to_vectors.values())

        # Record time and return final result
        time_taken = timer() - t0
        embeddings_final_result = {
            "content": content,
            "embeddings": embeddings,
            "processingTimeMs": round(time_taken * 1000)
        }
        logger.debug(f"embed request completed with total processing time: {(time_taken):.3f}s.")

        return embeddings_final_result




