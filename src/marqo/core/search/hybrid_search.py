import copy
import json
import typing
import uuid
from collections import defaultdict
from contextlib import ExitStack
from timeit import default_timer as timer
from typing import List, Optional, Union, Iterable, Sequence, Dict, Any, Tuple

import numpy as np
import psutil
from PIL import Image

import marqo.core.unstructured_vespa_index.common as unstructured_common
from marqo import marqo_docs
from marqo.api import exceptions as api_exceptions
from marqo.api import exceptions as errors
# We depend on _httprequests.py for now, but this may be replaced in the future, as
# _httprequests.py is designed for the client
from marqo.config import Config
from marqo.core import constants
from marqo.core import exceptions as core_exceptions
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models.marqo_index import IndexType
from marqo.core.models.marqo_index import MarqoIndex, FieldType, UnstructuredMarqoIndex, StructuredMarqoIndex
from marqo.core.models.marqo_query import MarqoTensorQuery, MarqoLexicalQuery, MarqoHybridQuery
from marqo.core.models.hybrid_parameters import HybridParameters
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
from marqo.tensor_search import utils, validation, add_docs
from marqo.tensor_search.enums import (
    Device, TensorField, SearchMethod, EnvVars
)
from marqo.tensor_search.index_meta_cache import get_cache
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from marqo.tensor_search.models.api_models import BulkSearchQueryEntity, ScoreModifierLists
from marqo.tensor_search.models.delete_docs_objects import MqDeleteDocsRequest
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.models.search import Qidx, JHash, SearchContext, VectorisedJobs, VectorisedJobPointer
from marqo.tensor_search.telemetry import RequestMetricsStore
from marqo.tensor_search.tensor_search_logging import get_logger
from marqo.vespa.exceptions import VespaStatusError
from marqo.vespa.models import VespaDocument, FeedBatchResponse, QueryResult
from marqo.tensor_search.tensor_search import run_vectorise_pipeline, gather_documents_from_response, logger
class HybridSearch:
    def search(
            self, config: Config, index_name: str, query: Optional[Union[str, dict]], result_count: int = 5,
            offset: int = 0, ef_search: Optional[int] = None, approximate: bool = True,
            searchable_attributes: Iterable[str] = None, filter_string: str = None, device: str = None,
            attributes_to_retrieve: Optional[List[str]] = None, boost: Optional[Dict] = None,
            image_download_headers: Optional[Dict] = None, context: Optional[Dict] = None,
            score_modifiers: Optional[ScoreModifierLists] = None, model_auth: Optional[ModelAuth] = None,
            highlights: bool = False, text_query_prefix: Optional[str] = None,
            hybrid_parameters: HybridParameters = None) -> Dict:
        """

            Args:
                config:
                index_name:
                query: either a string query (which can be a URL or natural language text), a dict of
                    <query string>:<weight float> pairs, or None with a context
                result_count:
                offset:
                searchable_attributes: Iterable of field names to search. If left as None, then all will
                    be searched
                verbose: if 0 - nothing is printed. if 1 - data is printed without vectors, if 2 - full
                    objects are printed out
                attributes_to_retrieve: if set, only returns these fields
                image_download_headers: headers for downloading images
                context: a dictionary to allow custom vectors in search
                score_modifiers: a dictionary to modify the score based on field values, for tensor search only
                model_auth: Authorisation details for downloading a model (if required)
                highlights: if True, highlights will be returned
            Returns:

            Note:
                - uses multisearch, which returns k results in each attribute. Not that much of a concern unless you have a
                ridiculous number of attributes
                - Should not be directly called by client - the search() method should
                be called. The search() method adds syncing
                - device should ALWAYS be set

            Output format:
                [
                    {
                        _id: doc_id
                        doc: {# original document},
                        highlights:[{}],
                    },
                ]
            Future work:
                - max result count should be in a config somewhere
                - searching a non existent index should return a HTTP-type error
            """

        # # SEARCH TIMER-LOGGER (pre-processing)
        if not device:
            raise api_exceptions.InternalError("_hybrid_search cannot be called without `device`!")

        RequestMetricsStore.for_request().start("search.hybrid.processing_before_vespa")

        marqo_index = index_meta_cache.get_index(config=config, index_name=index_name)

        # TODO: Remove when we support unstructured.
        if isinstance(marqo_index, UnstructuredMarqoIndex):
            raise core_exceptions.UnsupportedFeatureError("Unstructured indexes are not yet supported for hybrid search. "
                                                          "Please use a structured index.")

        # Determine the text query prefix
        text_query_prefix = marqo_index.model.get_text_query_prefix(text_query_prefix)

        queries = [BulkSearchQueryEntity(
            q=query, searchableAttributes=searchable_attributes, searchMethod=SearchMethod.HYBRID, limit=result_count,
            offset=offset, showHighlights=False, filter=filter_string, attributesToRetrieve=attributes_to_retrieve,
            boost=boost, image_download_headers=image_download_headers, context=context, scoreModifiers=score_modifiers,
            index=marqo_index, modelAuth=model_auth, text_query_prefix=text_query_prefix, hybridParameters=hybrid_parameters
        )]

        with RequestMetricsStore.for_request().time(f"search.hybrid.vector_inference_full_pipeline"):
            qidx_to_vectors: Dict[Qidx, List[float]] = run_vectorise_pipeline(config, queries, device)
        vectorised_text = list(qidx_to_vectors.values())[0]

        # Parse text into required and optional terms.
        (required_terms, optional_terms) = utils.parse_lexical_query(query)

        marqo_query = MarqoHybridQuery(
            index_name=index_name,
            vector_query=vectorised_text,
            filter=filter_string,
            limit=result_count,
            ef_search=ef_search,
            approximate=approximate,
            offset=offset,
            or_phrases=optional_terms,
            and_phrases=required_terms,
            attributes_to_retrieve=attributes_to_retrieve,
            searchable_attributes=searchable_attributes,
            score_modifiers=score_modifiers.to_marqo_score_modifiers() if score_modifiers is not None else None,
            # Hybrid-specific attributes
            score_modifiers_lexical=hybrid_parameters.score_modifiers_lexical.to_marqo_score_modifiers()
            if hybrid_parameters.score_modifiers_lexical is not None else None,
            score_modifiers_tensor=hybrid_parameters.score_modifiers_tensor.to_marqo_score_modifiers()
            if hybrid_parameters.score_modifiers_tensor is not None else None,
            hybrid_parameters=hybrid_parameters
        )

        vespa_index = vespa_index_factory(marqo_index)
        vespa_query = vespa_index.to_vespa_query(marqo_query)

        total_preprocess_time = RequestMetricsStore.for_request().stop("search.hybrid.processing_before_vespa")
        logger.debug(
            f"search (hybrid) pre-processing: took {(total_preprocess_time):.3f}ms to vectorize and process query.")

        # SEARCH TIMER-LOGGER (roundtrip)
        with RequestMetricsStore.for_request().time("search.hybrid.vespa",
                                                    lambda t: logger.debug(f"Vespa search: took {t:.3f}ms")
                                                    ):
            responses = config.vespa_client.query(**vespa_query)

        if not approximate and (responses.root.coverage.coverage < 100 or responses.root.coverage.degraded is not None):
            raise errors.InternalError(
                f'Graceful degradation detected for non-approximate search. '
                f'Coverage is not 100%: {responses.root.coverage}'
                f'Degraded: {str(responses.root.coverage.degraded)}'
            )

        # SEARCH TIMER-LOGGER (post-processing)
        RequestMetricsStore.for_request().start("search.hybrid.postprocess")
        gathered_docs = gather_documents_from_response(responses, marqo_index, highlights, attributes_to_retrieve)

        if boost is not None:
            raise api_exceptions.MarqoWebError('Boosting is not currently supported with Vespa')
            # gathered_docs = boost_score(gathered_docs, boost, searchable_attributes)

        # completely_sorted = sort_chunks(gathered_docs)

        # if verbose:
        #     print("Chunk vector search, sorted result:")
        #     if verbose == 1:
        #         pprint.pprint(utils.truncate_dict_vectors(completely_sorted))
        #     elif verbose == 2:
        #         pprint.pprint(completely_sorted)

        total_postprocess_time = RequestMetricsStore.for_request().stop("search.hybrid.postprocess")
        logger.debug(
            f"search (hybrid) post-processing: took {(total_postprocess_time):.3f}ms to sort and format "
            f"{len(gathered_docs)} results from Vespa."
        )

        return gathered_docs