"""tensor search logic

API Notes:
    - Some fields beginning with a double underscore "__" are protected and used for our internal purposes.
    - Examples include:
        __field_name
        __field_content
        __doc_chunk_relation
        __chunk_ids
        fields beginning with "__vector_"
    - The "_id" field isn't a real field. It's a way to declare an ID. Internally we use it as the ID
        for the doc. The doc is stored without this field in its body

Notes on search behaviour with caching and searchable attributes:
    The behaviour of lexical search and vector search differs when it comes to
    interactions between the cache and searchable attributes.

    This issue should just occur on the first search when another user adds a
    new field, as the index cache updates in the background during the search.

    Lexical search:
        - Searching an existing but uncached field will return the best result
            (the uncached field will be searched)
        - Searching all fields will return a poor result
            (the uncached field won't be searched)
    Vector search:
        - Searching an existing but uncached field will return no results (the
            uncached field won't be searched)
        - Searching all fields will return a poor result (the uncached field
            won't be searched)

"""
from collections import defaultdict

import psutil
import typing
from timeit import default_timer as timer
from typing import List, Optional, Union, Iterable, Sequence, Dict, Any, Tuple, Set

import marqo.core.inference.api.exceptions as inference_exceptions
from marqo import marqo_docs
from marqo.api import exceptions as api_exceptions
from marqo.api import exceptions as errors
from marqo.config import Config
from marqo.core import constants
from marqo.core import exceptions as core_exceptions
from marqo.core.inference.api import Modality, TextPreprocessingConfig, ImagePreprocessingConfig, \
    AudioPreprocessingConfig, VideoPreprocessingConfig, InferenceError, Inference, InferenceRequest, \
    EmbeddingModelConfig, \
    ModelError, InferenceErrorModel
from marqo.core.inference.modality_utils import infer_modality, is_base64_image
from marqo.core.models.facets_parameters import FacetsParameters
from marqo.core.models.hybrid_parameters import HybridParameters
from marqo.core.models.interpolation_method import InterpolationMethod
from marqo.core.models.marqo_get_documents_by_id_response import (MarqoGetDocumentsByIdsResponse,
                                                                  MarqoGetDocumentsByIdsItem)
from marqo.core.models.marqo_index import IndexType
from marqo.core.models.marqo_index import MarqoIndex
from marqo.core.models.marqo_index import SemiStructuredMarqoIndex
from marqo.core.models.marqo_query import MarqoTensorQuery, MarqoLexicalQuery
from marqo.core.structured_vespa_index import common as structured_common
from marqo.core.structured_vespa_index.common import RANK_PROFILE_BM25, RANK_PROFILE_EMBEDDING_SIMILARITY
from marqo.core.unstructured_vespa_index import common as unstructured_common
from marqo.core.utils.vector_interpolation import from_interpolation_method
from marqo.core.vespa_index.vespa_index import for_marqo_index as vespa_index_factory
from marqo.core.vespa_index.vespa_schema import MINIMUM_SEMI_STRUCTURED_INDEX_VERSION
from marqo.exceptions import InternalError
from marqo.logging import get_logger
from marqo.tensor_search import delete_docs
from marqo.tensor_search import index_meta_cache
from marqo.tensor_search import utils, validation
from marqo.tensor_search.enums import (
    Device, TensorField, SearchMethod
)
from marqo.tensor_search.enums import EnvVars
from marqo.tensor_search.index_meta_cache import get_cache
from marqo.tensor_search.models.api_models import BulkSearchQueryEntity, ScoreModifierLists
from marqo.tensor_search.models.api_models import CustomVectorQuery
from marqo.tensor_search.models.collapse_model import CollapseModel
from marqo.tensor_search.models.delete_docs_objects import MqDeleteDocsRequest
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.models.relevance_cutoff_model import RelevanceCutoffModel
from marqo.tensor_search.models.search import Qidx, JHash, SearchContext, VectorisedJobs, VectorisedJobPointer, \
    SearchContextTensor, QueryContentCollector, QueryContent
from marqo.tensor_search.models.sort_by_model import SortByModel
from marqo.tensor_search.telemetry import RequestMetricsStore
from marqo.tensor_search.utils import read_env_vars_and_defaults_ints
from marqo.vespa.exceptions import VespaStatusError
from marqo.vespa.models import QueryResult

logger = get_logger(__name__)


def _sanitize_query_for_response(query: Optional[Union[str, dict]]):
    """
    Replace base64 image content in queries with 'data:image/[omitted]' for response.
    
    Args:
        query: The query object which can be a string, dict, or CustomVectorQuery
        
    Returns:
        The sanitized query object with base64 content replaced
    """
    if query is None:
        return query
    
    if isinstance(query, str):
        if is_base64_image(query):
            return 'data:image/[omitted]'
        return query
    
    if isinstance(query, dict):
        sanitized_query = {}
        for key, value in query.items():
            if is_base64_image(key):
                sanitized_query['data:image/[omitted]'] = value
            else:
                sanitized_query[key] = value
        return sanitized_query

    # Should not reach here
    raise RuntimeError('Invalid query type')  # pragma: no cover

def _get_marqo_document_by_id(config: Config, index_name: str, document_id: str):
    marqo_index = _get_latest_index(config, index_name)

    try:
        res = config.vespa_client.get_document(document_id, marqo_index.schema_name)
    except VespaStatusError as e:
        if e.status_code == 404:
            raise api_exceptions.DocumentNotFoundError(
                f"Document with ID {document_id} not found in index {index_name}")
        else:
            raise e

    vespa_index = vespa_index_factory(marqo_index)
    marqo_document = vespa_index.to_marqo_document(res.document.dict())

    return marqo_document


def get_document_by_id(
        config: Config, index_name: str, document_id: str, show_vectors: bool = False):
    """returns document by its ID"""
    validation.validate_id(document_id)

    marqo_document = _get_marqo_document_by_id(config, index_name, document_id)

    if show_vectors:
        if constants.MARQO_DOC_TENSORS in marqo_document:
            marqo_document[TensorField.tensor_facets] = _get_tensor_facets(marqo_document[constants.MARQO_DOC_TENSORS])
        else:
            marqo_document[TensorField.tensor_facets] = []

    if not show_vectors:
        if unstructured_common.MARQO_DOC_MULTIMODAL_PARAMS in marqo_document:
            del marqo_document[unstructured_common.MARQO_DOC_MULTIMODAL_PARAMS]

    if constants.MARQO_DOC_TENSORS in marqo_document:
        del marqo_document[constants.MARQO_DOC_TENSORS]

    return marqo_document


def _get_marqo_documents_by_ids(
        config: Config, index_name: str, document_ids, ignore_invalid_ids: bool = False
):
    validated_ids = []
    for doc_id in document_ids:
        try:
            validated_ids.append(validation.validate_id(doc_id))
        except api_exceptions.InvalidDocumentIdError as e:
            if not ignore_invalid_ids:
                raise e
            logger.debug(f'Invalid document ID {doc_id} ignored')

    if len(validated_ids) == 0:  # Can only happen when ignore_invalid_ids is True
        return []

    marqo_index = _get_latest_index(config, index_name)
    batch_get = config.vespa_client.get_batch(validated_ids, marqo_index.schema_name)
    vespa_index = vespa_index_factory(marqo_index)

    return [vespa_index.to_marqo_document(response.document.dict()) for response in batch_get.responses
            if response.status == 200]


def get_documents_by_ids(
        config: Config, index_name: str, document_ids: typing.Collection[str],
        show_vectors: bool = False, ignore_invalid_ids: bool = False
) -> MarqoGetDocumentsByIdsResponse:
    """
    Returns documents by their IDs.

    Args:
        ignore_invalid_ids: If True, invalid IDs will be ignored and not returned in the response. If False, an error
            will be raised if any of the IDs are invalid
    """
    if not isinstance(document_ids, typing.Collection):
        raise api_exceptions.InvalidArgError("Get documents must be passed a collection of IDs!")
    if len(document_ids) <= 0:
        raise api_exceptions.InvalidArgError("Can't get empty collection of IDs!")

    max_docs_limit = utils.read_env_vars_and_defaults(EnvVars.MARQO_MAX_RETRIEVABLE_DOCS)
    if max_docs_limit is not None and len(document_ids) > int(max_docs_limit):
        raise api_exceptions.IllegalRequestedDocCount(
            f"{len(document_ids)} documents were requested, which is more than the allowed limit of [{max_docs_limit}], "
            f"set by the environment variable `{EnvVars.MARQO_MAX_RETRIEVABLE_DOCS}`")

    unsuccessful_docs: List[Tuple[int, MarqoGetDocumentsByIdsItem]] = []

    validated_ids = []
    for loc, doc_id in enumerate(document_ids):
        try:
            validated_ids.append(validation.validate_id(doc_id))
        except api_exceptions.InvalidDocumentIdError as e:
            if not ignore_invalid_ids:
                unsuccessful_docs.append(
                    (
                        loc, MarqoGetDocumentsByIdsItem(
                        # Invalid IDs are not returned in the response
                        id=doc_id,
                        message=e.message,
                        status=int(e.status_code)
                    )
                    )
                )
            else:
                logger.debug(f'Invalid document ID {doc_id} ignored')

    if len(validated_ids) == 0:  # Can only happen when ignore_invalid_ids is True
        return MarqoGetDocumentsByIdsResponse(errors=True, results=[i[1] for i in unsuccessful_docs])

    marqo_index = _get_latest_index(config, index_name)
    with RequestMetricsStore.for_request().time(f"get_documents.vespa"):
        batch_get = config.vespa_client.get_batch(validated_ids, marqo_index.schema_name)
    vespa_index = vespa_index_factory(marqo_index)

    results: List[Union[MarqoGetDocumentsByIdsItem, Dict]] = []
    errors = batch_get.errors

    for response in batch_get.responses:
        if response.status == 200:
            marqo_document = vespa_index.to_marqo_document(response.document.dict())
            if show_vectors:
                if constants.MARQO_DOC_TENSORS in marqo_document:
                    marqo_document[TensorField.tensor_facets] = _get_tensor_facets(
                        marqo_document[constants.MARQO_DOC_TENSORS])
                else:
                    marqo_document[TensorField.tensor_facets] = []

            if not show_vectors:
                if unstructured_common.MARQO_DOC_MULTIMODAL_PARAMS in marqo_document:
                    del marqo_document[unstructured_common.MARQO_DOC_MULTIMODAL_PARAMS]

            if constants.MARQO_DOC_TENSORS in marqo_document:
                del marqo_document[constants.MARQO_DOC_TENSORS]

            results.append(
                {
                    TensorField.found: True,
                    **marqo_document
                }
            )
        else:
            document = config.document
            status, message = document.vespa_client.translate_vespa_document_response(response.status, None)
            results.append(
                MarqoGetDocumentsByIdsItem(
                    id=_get_id_from_vespa_id(response.id), status=status,
                    found=False, message=message)
            )

    # Insert the error documents at the correct locations
    for loc, error_info in unsuccessful_docs:
        results.insert(loc, error_info)
        errors = True

    return MarqoGetDocumentsByIdsResponse(errors=errors, results=results)


def _get_latest_index(config: Config, index_name: str) -> MarqoIndex:
    """
    Get index from the cache first. If index is semi-structured, get the latest setting bypassing the cache
    This approach makes sure we don't add extra latency to structured indexes or legacy unstructured indexes since they
    never change. It also makes sure we always get the latest version of semi-structured index to guarantee the strong
    consistency.
    """
    marqo_index = index_meta_cache.get_index(index_management=config.index_management, index_name=index_name)
    if marqo_index.type == IndexType.SemiStructured:
        return config.index_management.get_index(index_name=index_name)
    return marqo_index


def _get_id_from_vespa_id(vespa_id: str) -> str:
    """Returns the document ID from a Vespa ID. Vespa IDs are of the form `namespace::document_id`."""
    return vespa_id.split('::')[-1]


def _get_tensor_facets(marqo_doc_tensors: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Reformat Marqo doc tensors field for API response
    """
    tensor_facets = []
    for tensor_field in marqo_doc_tensors:
        chunks = marqo_doc_tensors[tensor_field][constants.MARQO_DOC_CHUNKS]
        embeddings = marqo_doc_tensors[tensor_field][constants.MARQO_DOC_EMBEDDINGS]
        if len(chunks) != len(embeddings):
            raise api_exceptions.InternalError(
                f"Number of chunks ({len(chunks)}) and number of embeddings ({len(embeddings)}) "
                f"for field {tensor_field} must be the same.")

        for i in range(len(chunks)):
            tensor_facets.append(
                {
                    tensor_field: chunks[i],
                    TensorField.embedding: embeddings[i]
                }
            )

    return tensor_facets


def search(config: Config, index_name: str, text: Optional[Union[str, dict, CustomVectorQuery]],
           result_count: int = 3, offset: int = 0, rerank_depth: Optional[int] = None,
           highlights: bool = True, ef_search: Optional[int] = None,
           approximate: Optional[bool] = None, approximate_threshold: Optional[float] = None,
           search_method: Union[str, SearchMethod, None] = SearchMethod.TENSOR,
           searchable_attributes: Iterable[str] = None, verbose: int = 0,
           reranker: Union[str, Dict] = None, filter: Optional[str] = None,
           attributes_to_retrieve: Optional[List[str]] = None,
           device: str = None, boost: Optional[Dict] = None,
           media_download_headers: Optional[Dict] = None,
           context: Optional[SearchContext] = None,
           score_modifiers: Optional[ScoreModifierLists] = None,
           model_auth: Optional[ModelAuth] = None,
           processing_start: float = None,
           text_query_prefix: Optional[str] = None,
           hybrid_parameters: Optional[HybridParameters] = None,
           facets: Optional[FacetsParameters] = None,
           track_total_hits: Optional[bool] = None,
           language: Optional[str] = None,
           relevance_cutoff: Optional[RelevanceCutoffModel] = None,
           sort_by: Optional[SortByModel] = None,
           interpolation_method: Optional[InterpolationMethod] = None,
           collapse: Optional[CollapseModel] = None,
           recency_parameters=None
           ) -> Dict:
    """The root search method. Calls the specific search method

    Validation should go here. Validations include:
        - all args and their types
        - result_count (negatives etc)
        - text

    This deals with index caching

    Args:
        config:
        index_name:
        text:
        result_count:
        offset:
        rerank_depth:
        search_method:
        searchable_attributes:
        verbose:
        device: May be none, we calculate default device here
        num_highlights: number of highlights to return for each doc
        boost: boosters to re-weight the scores of individual fields
        media_download_headers: headers to use when downloading media
        context: a dictionary to allow custom vectors in search, for tensor search only
        score_modifiers: a dictionary to modify the score based on field values, for tensor search only
        model_auth: Authorisation details for downloading a model (if required)
        text_query_prefix: The prefix to be used for chunking text fields or search queries.
        hybrid_parameters: Parameters for hybrid search
        facets: Parameters for facets
        track_total_hits: Whether to track total hits for the search
        interpolation_method: The interpolation method to use for the combining of vectors
    Returns:

    """

    # Validation for: result_count (limit) & offset
    # Validate neither is negative
    if result_count <= 0 or (not isinstance(result_count, int)):
        raise errors.IllegalRequestedDocCount(
            f"result_count must be an integer greater than 0! Received {result_count}"
        )

    if offset < 0:
        raise api_exceptions.IllegalRequestedDocCount("search result offset cannot be less than 0!")

        # validate query
    validation.validate_query(q=text, search_method=search_method)

    # Validate max limits
    max_docs_limit = utils.read_env_vars_and_defaults(EnvVars.MARQO_MAX_RETRIEVABLE_DOCS)
    max_search_limit = utils.read_env_vars_and_defaults(EnvVars.MARQO_MAX_SEARCH_LIMIT)
    max_search_offset = utils.read_env_vars_and_defaults(EnvVars.MARQO_MAX_SEARCH_OFFSET)
    max_search_context_docs = utils.read_env_vars_and_defaults(EnvVars.MARQO_MAX_SEARCH_CONTEXT_DOCS)

    check_upper = True if max_docs_limit is None else result_count + offset <= int(max_docs_limit)
    check_limit = True if max_search_limit is None else result_count <= int(max_search_limit)
    check_offset = True if max_search_offset is None else offset <= int(max_search_offset)
    if not check_upper:
        upper_bound_explanation = ("The search result limit + offset must be less than or equal to the "
                                   f"MARQO_MAX_RETRIEVABLE_DOCS limit of [{max_docs_limit}]. ")

        raise api_exceptions.IllegalRequestedDocCount(
            f"{upper_bound_explanation} Marqo received search result limit of `{result_count}` "
            f"and offset of `{offset}`.")
    if not check_limit:
        raise api_exceptions.IllegalRequestedDocCount(
            f"The search result limit must be less than or equal to the MARQO_MAX_SEARCH_LIMIT limit of "
            f"[{max_search_limit}]. Marqo received search result limit of `{result_count}`.")
    if not check_offset:
        raise api_exceptions.IllegalRequestedDocCount(
            f"The search result offset must be less than or equal to the MARQO_MAX_SEARCH_OFFSET limit of "
            f"[{max_search_offset}]. Marqo received search result offset of `{offset}`.")

    if processing_start is None:
        t0 = timer()
    else:
        t0 = processing_start

    validation.validate_context(context=context, query=text, search_method=search_method)
    validation.validate_boost(boost=boost, search_method=search_method)
    validation.validate_searchable_attributes(searchable_attributes=searchable_attributes, search_method=search_method)
    if searchable_attributes is not None:
        [validation.validate_field_name(attribute) for attribute in searchable_attributes]
    if attributes_to_retrieve is not None:
        [validation.validate_field_name(attribute) for attribute in attributes_to_retrieve]
    if verbose:
        print(f"determined_search_method: {search_method}, text query: {text}")

    selected_device = device

    # Fetch marqo index to pass to search method
    marqo_index = index_meta_cache.get_index(index_management=config.index_management, index_name=index_name)
    marqo_index_version = marqo_index.parsed_marqo_version()
    
    # Validate collapse field configuration
    if collapse is not None:
        # Validate if the index version support this feature
        if (marqo_index_version < constants.MARQO_COLLAPSE_FIELDS_MINIMUM_VERSION or
                not isinstance(marqo_index, SemiStructuredMarqoIndex)):
            index_type = 'structured' if marqo_index.type == IndexType.Structured else 'unstructured'
            raise core_exceptions.UnsupportedFeatureError(
                f"The 'collapseFields' search parameter is only supported for unstructured indexes created with "
                f"Marqo version {str(constants.MARQO_COLLAPSE_FIELDS_MINIMUM_VERSION)} or later. "
                f"This index is {index_type} and was created with Marqo {marqo_index_version}."
            )

        # Validate collapse field exists in index configuration
        if not marqo_index.is_collapse_field(collapse.name):
            raise api_exceptions.InvalidArgError(f"Field '{collapse.name}' is not configured as a collapse field "
                                                 f"for this index")
    
    if rerank_depth is not None \
            and marqo_index_version < constants.MARQO_RERANK_DEPTH_MINIMUM_VERSION:
        raise core_exceptions.UnsupportedFeatureError(
            f"The 'rerankDepth' search parameter is only supported for indexes created with Marqo version "
            f"{str(constants.MARQO_RERANK_DEPTH_MINIMUM_VERSION)} or later. "
            f"This index was created with Marqo {marqo_index_version}."
        )

    if search_method.upper() in {SearchMethod.TENSOR, SearchMethod.HYBRID}:
        # Default approximate and efSearch -- we can't set these at API-level since they're not a valid args
        # for lexical search
        if approximate is None:
            approximate = True

        # Add context.documents exclusion filter to exclude input docs (only applicable for tensor & hybrid)
        if context is not None and context.documents is not None:
            # Disallow context docs for legacy unstructured indexes
            if marqo_index.type == IndexType.Unstructured:
                raise core_exceptions.UnsupportedFeatureError(
                    f"Search context is not supported for unstructured indexes created with Marqo version "
                    f"{MINIMUM_SEMI_STRUCTURED_INDEX_VERSION} or later. "
                    f"This index was created with Marqo {marqo_index_version}."
                )
            if len(context.documents.ids) > int(max_search_context_docs):
                raise api_exceptions.IllegalRequestedDocCount(
                    f"Search context documents limit exceeded. "
                    f"Maximum allowed is {max_search_context_docs}, but got {len(context.documents.ids)}. "
                    f"To increase, set the environment variable '{EnvVars.MARQO_MAX_SEARCH_CONTEXT_DOCS}'"
                )
            if context.documents.parameters.exclude_input_documents:
                filter = config.recommender.get_exclusion_filter(marqo_index, list(context.documents.ids.keys()), filter)

        if search_method.upper() == SearchMethod.TENSOR:
            search_result = _vector_text_search(
                config=config, marqo_index=marqo_index, query=text, result_count=result_count, offset=offset,
                ef_search=ef_search, approximate=approximate, approximate_threshold=approximate_threshold,
                searchable_attributes=searchable_attributes,
                filter_string=filter, device=selected_device, attributes_to_retrieve=attributes_to_retrieve,
                boost=boost,
                media_download_headers=media_download_headers, context=context, score_modifiers=score_modifiers,
                model_auth=model_auth, highlights=highlights, text_query_prefix=text_query_prefix, 
                rerank_depth=rerank_depth, interpolation_method=interpolation_method
            )
        else:  # SearchMethod.HYBRID
            # TODO: Deal with circular import when all modules are refactored out.
            from marqo.core.search.hybrid_search import HybridSearch
            search_result = HybridSearch().search(
                config=config, marqo_index=marqo_index, query=text, result_count=result_count, offset=offset,
                rerank_depth=rerank_depth,
                ef_search=ef_search, approximate=approximate, approximate_threshold=approximate_threshold,
                searchable_attributes=searchable_attributes,
                filter_string=filter, device=selected_device, attributes_to_retrieve=attributes_to_retrieve,
                boost=boost,
                media_download_headers=media_download_headers, context=context, score_modifiers=score_modifiers,
                model_auth=model_auth, highlights=highlights, text_query_prefix=text_query_prefix,
                hybrid_parameters=hybrid_parameters, facets=facets, track_total_hits=track_total_hits,
                language=language,
                relevance_cutoff=relevance_cutoff, sort_by=sort_by,
                interpolation_method=interpolation_method,
                collapse=collapse,
                recency_parameters=recency_parameters
            )

    elif search_method.upper() == SearchMethod.LEXICAL:
        if ef_search is not None:
            raise errors.InvalidArgError(
                f"efSearch is not a valid argument for lexical search")
        if approximate is not None:
            raise errors.InvalidArgError(
                f"approximate is not a valid argument for lexical search")

        search_result = _lexical_search(
            config=config, marqo_index=marqo_index, text=text, result_count=result_count, offset=offset,
            searchable_attributes=searchable_attributes, verbose=verbose,
            filter_string=filter, attributes_to_retrieve=attributes_to_retrieve, highlights=highlights,
            score_modifiers=score_modifiers, language=language
        )
    else:
        raise api_exceptions.InvalidArgError(f"Search called with unknown search method: {search_method}")

    if reranker is not None:
        raise api_exceptions.InvalidArgError(f"Reranker is no longer supported in Marqo version 2.17 and later")

    if isinstance(text, CustomVectorQuery):
        search_result["query"] = text.dict()  # Make object JSON serializable
    else:
        search_result["query"] = _sanitize_query_for_response(text)

    search_result["limit"] = result_count
    search_result["offset"] = offset

    time_taken = timer() - t0
    search_result["processingTimeMs"] = round(time_taken * 1000)
    logger.debug(f"search ({search_method.lower()}) completed with total processing time: {(time_taken):.3f}s.")

    return search_result


def _lexical_search(
        config: Config, marqo_index: MarqoIndex, text: str, result_count: int = 3, offset: int = 0,
        searchable_attributes: Sequence[str] = None, verbose: int = 0, filter_string: str = None,
        highlights: bool = True, attributes_to_retrieve: Optional[List[str]] = None, expose_facets: bool = False,
        score_modifiers: Optional[ScoreModifierLists] = None, language: Optional[str] = None):
    """

    Args:
        config:
        marqo_index: index object fetched by calling function
        text:
        result_count:
        offset:
        searchable_attributes:
        verbose:

    Returns:

    Notes:
        Should not be directly called by client - the search() method should
        be called. The search() method adds syncing
        Uses normal search (not multiple search).
    TODO:
        - Test raise_for_searchable_attribute=False
    """
    if not isinstance(text, str):
        raise api_exceptions.InvalidArgError(
            f"Query arg must be of type str! text arg is of type {type(text)}. "
            f"Query arg: {text}")

    # SEARCH TIMER-LOGGER (pre-processing)
    RequestMetricsStore.for_request().start("search.lexical.processing_before_vespa")

    index_name = marqo_index.name

    # Parse text into required and optional terms.
    (required_terms, optional_terms) = utils.parse_lexical_query(text)

    marqo_query = MarqoLexicalQuery(
        index_name=index_name,
        or_phrases=optional_terms,
        and_phrases=required_terms,
        filter=filter_string,
        limit=result_count,
        offset=offset,
        searchable_attributes=searchable_attributes,
        attributes_to_retrieve=attributes_to_retrieve,
        score_modifiers=score_modifiers.to_marqo_score_modifiers() if score_modifiers else None,
        language=language
    )

    vespa_index = vespa_index_factory(marqo_index)
    vespa_query = vespa_index.to_vespa_query(marqo_query)

    total_preprocess_time = RequestMetricsStore.for_request().stop("search.lexical.processing_before_vespa")
    logger.debug(f"search (lexical) pre-processing: took {(total_preprocess_time):.3f}ms to process query.")

    with RequestMetricsStore.for_request().time("search.lexical.vespa",
                                                lambda t: logger.debug(f"Vespa search: took {t:.3f}ms")
                                                ):
        try:
            responses = config.vespa_client.query(**vespa_query)
        except VespaStatusError as e:
            # The index will not have the bm25 rank profile if there are no lexical fields
            if f"does not contain requested rank profile '{RANK_PROFILE_BM25}'" in e.message:
                raise core_exceptions.InvalidArgumentError(
                    f"Index {index_name} has no lexically searchable fields, thus lexical search cannot be performed. "
                    f"Please create an index with a lexically searchable field, or try a different search method."
                )
            raise e

    # SEARCH TIMER-LOGGER (post-processing)
    RequestMetricsStore.for_request().start("search.lexical.postprocess")
    gathered_docs = gather_documents_from_response(responses, marqo_index, False, attributes_to_retrieve)

    # Set the _highlights for each doc as [] to follow Marqo-V1's convention
    if highlights:
        for docs in gathered_docs['hits']:
            docs['_highlights'] = []

    total_postprocess_time = RequestMetricsStore.for_request().stop("search.lexical.postprocess")
    logger.debug(
        f"search (lexical) post-processing: took {(total_postprocess_time):.3f}ms to format "
        f"{len(gathered_docs)} results."
    )

    return gathered_docs


def construct_vector_input_batches(query: Optional[Union[str, Dict]], media_download_headers: Optional[Dict] = None) \
        -> QueryContentCollector:
    """Splits images from text in a single query (either a query string, or dict of weighted strings).

    Args:
        query: a string query, or a dict of weighted strings.
        media_download_headers: headers to use when downloading media

    Returns:
        A SearchQueryCollector object with the text and media content separated.
    """
    # TODO - infer this from model
    query_content_list = []
    if isinstance(query, str):
        query_content_list.append(
            QueryContent(
                content=query,
                modality=infer_modality(query, media_download_headers=media_download_headers)
            )
        )
    elif isinstance(query, dict):  # is dict:
        for query, weights in query.items():
            query_content_list.append(
                QueryContent(
                    content=query,
                    modality=infer_modality(query, media_download_headers=media_download_headers)
                )
            )
    elif query is None:
        pass
    else:
        raise ValueError(f"Incorrect type for query: {type(query).__name__}")
    return QueryContentCollector(queries=query_content_list)


def gather_documents_from_response(response: QueryResult, marqo_index: MarqoIndex, highlights: bool,
                                   attributes_to_retrieve: List[str] = None) -> Dict[str, Any]:
    """
    Convert a VespaQueryResponse to a Marqo search response
    """

    if (marqo_index.type in [IndexType.Unstructured, IndexType.SemiStructured] and
            attributes_to_retrieve is not None):
        # Unstructured index and Semi-structured index stores fixed fields (numeric, boolean, string arrays, etc.) in
        # combined field. It needs to select attributes after converting vespa doc to marqo doc if
        # attributes_to_retrieve is specified
        metadata_fields_to_retrieve = {"_id", "_score", "_highlights"}
        attributes_to_retrieve_set = set(attributes_to_retrieve).union(metadata_fields_to_retrieve)
    else:
        # If this set is None, we will return the marqo_doc as is.
        attributes_to_retrieve_set = None

    vespa_index = vespa_index_factory(marqo_index)
    hits = []
    for doc in response.hits:
        if doc.id.startswith("group:facet:"):  # Not an actual document id but group's id returned by vespa
            continue
        marqo_doc = vespa_index.to_marqo_document(dict(doc), return_highlights=highlights)
        marqo_doc["_score"] = doc.relevance

        if attributes_to_retrieve_set is not None:
            marqo_doc = select_attributes(marqo_doc, attributes_to_retrieve_set)

        # Delete chunk data
        if constants.MARQO_DOC_TENSORS in marqo_doc:
            del marqo_doc[constants.MARQO_DOC_TENSORS]
        hits.append(marqo_doc)

    return {'hits': hits}


def select_attributes(marqo_doc: Dict[str, Any], attributes_to_retrieve_set: Set[str]) -> Dict[str, Any]:
    """
    Unstructured index and Semi-structured index retrieve all fixed fields (numeric, boolean, string arrays, etc.)
    from Vespa when attributes_to_retrieve is specified. After converting the Vespa doc to Marqo doc, it needs to
    filter out attributes not in the attributes_to_retrieve list.

    Please note that numeric map fields are flattened for unstructured or semi-structured indexes.
    Therefore, when filtering on attributes_to_retrieve, we need to also include flattened map fields
    with the specified attributes as prefixes. We keep this behaviour only for compatibility reasons.
    """
    return {k: v for k, v in marqo_doc.items() if k in attributes_to_retrieve_set or
            '.' in k and k.split('.', maxsplit=1)[0] in attributes_to_retrieve_set}


def assign_query_to_vector_job(
        q: BulkSearchQueryEntity, jobs: Dict[JHash, VectorisedJobs],
        grouped_content: QueryContentCollector,
        index_info: MarqoIndex, device: str) -> List[VectorisedJobPointer]:
    """
    For a individual query, assign its content (to be vectorised) to a vector job. If none exist with the correct
    specifications, create a new job.

    Mutates entries in, and adds values to, the `jobs` param.

    Args:
        q:
        jobs:
        grouped_content: a 2-tuple of content, belonging to a single query, the first element is a list of text content.
            The second is a list of image URLs. Either element can be an empty list
        index_info:
        device:

    Returns:
        A list of pointers to the location in a vector job that will have its vectorised content.
    """
    ptrs = []
    content_lists_by_modality = [
        grouped_content.text_queries,
        grouped_content.image_queries,
        grouped_content.audio_queries,
        grouped_content.video_queries,
    ]

    for i, list_of_queries_by_modalities in enumerate(content_lists_by_modality):
        if len(list_of_queries_by_modalities) > 0:
            content: List[str] = [query.content for query in list_of_queries_by_modalities]
            modality: Modality = list_of_queries_by_modalities[0].modality
            vector_job = VectorisedJobs(
                model_name=index_info.model.name,
                model_properties=index_info.model.get_properties(),
                content=content,
                device=device,
                normalize_embeddings=index_info.normalize_embeddings,
                media_download_headers=q.mediaDownloadHeaders,
                model_auth=q.modelAuth,
                modality=modality
            )
            # If exists, add content to vector job. Otherwise create new
            if jobs.get(vector_job.groupby_key()) is not None:
                j = jobs.get(vector_job.groupby_key())
                ptrs.append(j.add_content(content))
            else:
                jobs[vector_job.groupby_key()] = vector_job
                ptrs.append(VectorisedJobPointer(
                    job_hash=vector_job.groupby_key(),
                    start_idx=0,
                    end_idx=len(vector_job.content)
                ))
    return ptrs


def create_vector_jobs(queries: List[BulkSearchQueryEntity], config: Config, device: str) -> Tuple[
    Dict[Qidx, List[VectorisedJobPointer]], Dict[JHash, VectorisedJobs]]:
    """
        For each query:
            - Find what needs to be vectorised
            - Group content (across search requests), that could be vectorised together
            - Keep track of the Job related to a search query

        Returns:
            - A mapping of the query index to the VectorisedJobPointer that points to the VectorisedJobs that will process its content.
            - A mapping of job key to job (for fast access).
    """
    qidx_to_job: Dict[Qidx, List[VectorisedJobPointer]] = dict()
    jobs: Dict[JHash, VectorisedJobs] = {}
    for i, q in enumerate(queries):
        # split images, from text:
        to_be_vectorised: QueryContentCollector = construct_vector_input_batches(q.q, q.mediaDownloadHeaders)
        qidx_to_job[i] = assign_query_to_vector_job(q, jobs, to_be_vectorised, q.index, device)

    return qidx_to_job, jobs


def _get_preprocessing_config(modality: Modality, media_download_headers: Optional[Dict[str, str]]):
    """
    Get the preprocessing config for the given modality used for searching.
    """
    if modality == Modality.TEXT:
        return TextPreprocessingConfig()  # the prefix has been added to the query, so we don't need to specify it here
    elif modality == Modality.IMAGE:
        return ImagePreprocessingConfig(download_header=media_download_headers, download_thread_count=1)
    elif modality == Modality.AUDIO:
        return AudioPreprocessingConfig(
            download_header=media_download_headers, download_thread_count=1,
            max_media_size_bytes=read_env_vars_and_defaults_ints(EnvVars.MARQO_MAX_SEARCH_VIDEO_AUDIO_FILE_SIZE)
        )
    elif modality == Modality.VIDEO:
        return VideoPreprocessingConfig(
            download_header=media_download_headers, download_thread_count=1,
            max_media_size_bytes=read_env_vars_and_defaults_ints(EnvVars.MARQO_MAX_SEARCH_VIDEO_AUDIO_FILE_SIZE)
        )
    else:
        raise InferenceError(f'Unsupported modality: {modality}')


def vectorise_jobs(inference: Inference, jobs: List[VectorisedJobs]) -> Dict[JHash, Dict[str, List[float]]]:
    """ Run inference.vectorise() against each vector jobs."""
    result: Dict[JHash, Dict[str, List[float]]] = dict()
    for v in jobs:
        if not v.content:
            continue
        try:
            inference_request = InferenceRequest(
                modality=v.modality,
                contents=v.content,
                embedding_model_config=EmbeddingModelConfig(
                    model_name=v.model_name,
                    model_properties=v.model_properties,
                    model_auth=v.model_auth,
                    normalize_embeddings=v.normalize_embeddings,
                ),
                device=v.device,
                use_inference_cache=True,
                return_individual_error=False,
                preprocessing_config=_get_preprocessing_config(v.modality, v.media_download_headers),
            )

            inference_result = inference.vectorise(inference_request)

            # Sanity check the response from Inference
            if len(inference_result.result) != len(v.content):
                raise InternalError(f'Inference result contains embeddings for {len(inference_result.result)} '
                                    f'query items, but {len(v.content)} is expected')
            individual_errors = [f'{v.content[index]}: {r.error_message}'
                                 for index, r in enumerate(inference_result.result)
                                 if isinstance(r, InferenceErrorModel)]
            if individual_errors:
                raise InternalError(f'Individual errors returned when vectorising query string: {individual_errors}')
            chunked_contents = [(v.content[index], len(chunks)) for index, chunks in enumerate(inference_result.result)
                                if len(chunks) > 1]
            if chunked_contents:
                raise InternalError(f'Tensor query string should not be chunked but some '
                                    f'query items have multiple chunks: {chunked_contents}')

            # The per_content_result format is [('chunk', np.array())]
            vectors = [per_content_result[0][1].tolist() for per_content_result in inference_result.result]
            result[v.groupby_key()] = dict(zip(v.content, vectors))

        except ModelError as e:
            raise api_exceptions.BadRequestError(
                message=f'Problem vectorising query. Reason: {str(e)}',
                link=marqo_docs.list_of_models()
            ) from e

        except InferenceError as e:
            # TODO: differentiate image processing errors from other types of vectorise errors
            raise api_exceptions.InvalidArgError(message=f'Error vectorising content: {v.content}. '
                                                         f'Message: {e.message}') from e
    return result


def get_query_vectors_from_jobs(
        queries: List[BulkSearchQueryEntity], qidx_to_job: Dict[Qidx, List[VectorisedJobPointer]],
        job_to_vectors: Dict[JHash, Dict[str, List[float]]], config: Config,
        jobs: Dict[JHash, VectorisedJobs], interpolation_method: Optional[InterpolationMethod] = None,
) -> Dict[Qidx, List[float]]:
    """
    Retrieve the vectorised content associated to each query from the set of batch vectorise jobs.
    Handles multi-modal queries, by weighting and combining queries into a single vector.

    Args:
        - queries: Original search queries.
        - qidx_to_job: VectorisedJobPointer for each query
        - job_to_vectors: inference output from each VectorisedJob
        - config: standard Marqo config.

    Raises:
        api_exceptions.InvalidArgError: If this method can not collect a valid vector from the query
    """
    result: Dict[Qidx, List[float]] = defaultdict(list)
    for qidx, ptrs in qidx_to_job.items():

        # vectors = job_to_vectors[ptrs.job_hash][ptrs.start_idx: ptrs.end_idx]

        # qidx_to_vectors[qidx].append(vectors)
        q = queries[qidx]

        if isinstance(q.q, dict) or q.q is None:
            ordered_queries = list(q.q.items()) if isinstance(q.q, dict) else None
            # Store weights and vectors separately for use in interpolation
            collected_weights: List[List[float]] = []
            collected_vectors: List[float] = []

            if ordered_queries:
                # multiple queries. We have to weight and combine them:
                vectorised_ordered_queries = [
                    (
                        get_content_vector(
                            possible_jobs=qidx_to_job[qidx],
                            job_to_vectors=job_to_vectors,
                            content=content
                        ),
                        weight,
                        content
                    ) for content, weight in ordered_queries
                ]
                # TODO how do we ensure order?
                collected_weights = [weight for _, weight, _ in vectorised_ordered_queries]
                collected_vectors = [vec for vec, _, _ in vectorised_ordered_queries]

            # Add context tensors
            context_tensors = q.get_context_tensor()
            if context_tensors is not None:
                collected_weights += [v.weight for v in context_tensors]
                collected_vectors += [v.vector for v in context_tensors]

            # Add context document vectors
            context_documents = q.get_context_documents()
            if interpolation_method is None:
                interpolation_method = config.recommender.get_default_interpolation_method(q.index, context_documents)

            if context_documents:
                with RequestMetricsStore.for_request().time(f"search.vectorise.get_doc_vectors_from_ids"):
                                            context_doc_vectors = config.recommender.get_doc_vectors_from_ids(
                            index_name=q.index.name,
                            documents=context_documents.ids,
                            tensor_fields=context_documents.parameters.tensor_fields,
                            allow_missing_documents=context_documents.parameters.allow_missing_documents,
                            allow_missing_embeddings= context_documents.parameters.allow_missing_embeddings
                        )

                # Update weights and vectors list
                for document_id, vector_list in context_doc_vectors.items():
                    weight = context_documents.ids[document_id]
                    # Per doc, add whole list of vectors, copy the doc weight for each
                    collected_vectors.extend(vector_list)
                    collected_weights.extend([weight] * len(vector_list))

                # Save original doc ids for exclusion filtering
                all_document_ids = list(context_documents.ids.keys())

            # Make sure all vectors are the same size
            for vector in collected_vectors:
                if not q.index.model.get_dimension() == len(vector):
                    raise api_exceptions.InvalidArgError(
                        f"The dimension of the vectors returned by the model or given by the context vectors "
                        f"does not match the expected dimension. "
                        f"Expected dimension {q.index.model.get_dimension()} but got {len(vector)}"
                    )

            # Use interpolation to combine all vectors
            vector_interpolation = from_interpolation_method(interpolation_method)
            with RequestMetricsStore.for_request().time(f"search.vectorise.interpolate_vectors"):
                if collected_vectors:
                    merged_vector = vector_interpolation.interpolate(
                        vectors=collected_vectors,
                        weights=collected_weights
                    )
                    result[qidx] = list(merged_vector)
                else:
                    result[qidx] = []
        elif isinstance(q.q, str):
            if q.context:
                raise core_exceptions.InvalidArgumentError(
                    f"Cannot use 'context' for a search with a string 'q' (or queryTensor): '{q.q}'. "
                    f"To use 'context', please provide a dictionary or a CustomVectorQuery object as the query instead."
                )
            result[qidx] = get_content_vector(
                possible_jobs=qidx_to_job.get(qidx, []),
                job_to_vectors=job_to_vectors,
                content=q.q
            )
        else:
            raise ValueError(f"Unexpected query type: {type(q.q).__name__}")

        if not result[qidx]:
            raise api_exceptions.InvalidArgError(
                f"Marqo could not collect any vectors from the search query but the retrieval or ranking method requires "
                f"at least one valid vector. "
                f"Please check the provided query, context (if any), or queryTensor(for Hybrid search) "
            )

    return result


def get_content_vector(
        possible_jobs: List[VectorisedJobPointer],
        job_to_vectors: Dict[JHash, Dict[str, List[float]]],
        content: str
) -> List[float]:
    """finds the vector associated with a piece of content

    Args:
        possible_jobs: The jobs where the target vector may reside
        job_to_vectors: The mapping of job to vectors
        content: The content to search

    Returns:
        Associated vector, if it is found.

    Raises runtime error if is not found
    """
    not_found_error = RuntimeError(f"get_content_vector(): could not find corresponding vector for content `{content}`")
    for vec_job_pointer in possible_jobs:
        if content in job_to_vectors[vec_job_pointer.job_hash]:
            return job_to_vectors[vec_job_pointer.job_hash][content]
    raise not_found_error


def add_prefix_to_queries(queries: List[BulkSearchQueryEntity]) -> List[BulkSearchQueryEntity]:
    """
    Add prefix to the queries if it is a text query.

    Raises:
        MediaDownloadError: If the media cannot be downloaded
    """
    prefixed_queries = []
    for q in queries:
        text_query_prefix = q.index.model.get_text_query_prefix(q.text_query_prefix)

        if q.q is None:
            prefixed_q = q.q
        elif isinstance(q.q, str):
            modality = infer_modality(q.q, q.mediaDownloadHeaders)
            if modality == Modality.TEXT:
                prefixed_q = f"{text_query_prefix}{q.q}"
            else:
                prefixed_q = q.q
        else:  # q.q is dict
            prefixed_q = {}
            for key, value in q.q.items():
                # Apply prefix if key is not an image or if index does not treat URLs and pointers as images
                modality = infer_modality(key, q.mediaDownloadHeaders)
                if modality == Modality.TEXT:
                    prefixed_q[f"{text_query_prefix}{key}"] = value
                else:
                    prefixed_q[key] = value
        new_query_object = BulkSearchQueryEntity(
            q=prefixed_q,
            searchableAttributes=q.searchableAttributes,
            searchMethod=q.searchMethod,
            limit=q.limit,
            offset=q.offset,
            showHighlights=q.showHighlights,
            filter=q.filter,
            attributesToRetrieve=q.attributesToRetrieve,
            boost=q.boost,
            mediaDownloadHeaders=q.mediaDownloadHeaders,
            context=q.context,
            scoreModifiers=q.scoreModifiers,
            index=q.index,
            modelAuth=q.modelAuth,
            text_query_prefix=q.text_query_prefix,
            hybridParameters=q.hybridParameters
        )
        prefixed_queries.append(new_query_object)

    return prefixed_queries


def run_vectorise_pipeline(config: Config, queries: List[BulkSearchQueryEntity], device: Union[Device, str],
                           interpolation_method: InterpolationMethod = None) -> Dict[
    Qidx, List[float]]:
    """Run the query vectorisation process. This is a pipeline used for both Tensor search and Hybrid search.

    Raise:
        api_exceptions.InvalidArgError: If the vectorisation process fails or if the media cannot be downloaded.
    """

    # Prepend the prefixes to the queries if it exists (output should be of type List[BulkSearchQueryEntity])
    try:
        prefixed_queries = add_prefix_to_queries(queries)
    except inference_exceptions.MediaDownloadError as e:
        raise api_exceptions.InvalidArgError(message=str(e)) from e

    # 1. Pre-process inputs ready for s2_inference.vectorise
    # we can still use qidx_to_job. But the jobs structure may need to be different
    vector_jobs_tuple: Tuple[Dict[Qidx, List[VectorisedJobPointer]], Dict[JHash, VectorisedJobs]] = create_vector_jobs(
        prefixed_queries, config, device)

    qidx_to_jobs, jobs = vector_jobs_tuple

    # 2. Vectorise in batches against all queries
    ## TODO: To ensure that we are vectorising in batches, we can mock vectorise (), and see if the number of calls is as expected (if batch_size = 16, and number of docs = 32, and all args are the same, then number of calls = 2)
    # TODO: we need to enable str/PIL image structure:
    with RequestMetricsStore.for_request().time(f"search.vector.inference.vectorise_jobs"):
        job_ptr_to_vectors: Dict[JHash, Dict[str, List[float]]] = vectorise_jobs(config.inference, list(jobs.values()))

    # 3. For each query, get associated vectors
    # Combination of context tensors & documents is also done here
    qidx_to_vectors: Dict[Qidx, List[float]] = get_query_vectors_from_jobs(
        prefixed_queries, qidx_to_jobs, job_ptr_to_vectors, config, jobs, interpolation_method
    )
    return qidx_to_vectors


def _vector_text_search(
        config: Config, marqo_index: MarqoIndex,
        query: Optional[Union[str, dict, CustomVectorQuery]], result_count: int = 5,
        offset: int = 0,
        ef_search: Optional[int] = None, approximate: bool = True, approximate_threshold: Optional[float] = None,
        searchable_attributes: Iterable[str] = None, filter_string: str = None, device: str = None,
        attributes_to_retrieve: Optional[List[str]] = None, boost: Optional[Dict] = None,
        media_download_headers: Optional[Dict] = None, context: Optional[SearchContext] = None,
        score_modifiers: Optional[ScoreModifierLists] = None, model_auth: Optional[ModelAuth] = None,
        highlights: bool = False, text_query_prefix: Optional[str] = None, rerank_depth: Optional[int] = None,
        interpolation_method: Optional[InterpolationMethod] = None
) -> Dict:
    """

    Args:
        config:
        marqo_index: index object fetched by calling function
        query: either a string query (which can be a URL or natural language text), a dict of
            <query string>:<weight float> pairs, or None with a context
        result_count:
        offset:
        searchable_attributes: Iterable of field names to search. If left as None, then all will
            be searched
        verbose: if 0 - nothing is printed. if 1 - data is printed without vectors, if 2 - full
            objects are printed out
        attributes_to_retrieve: if set, only returns these fields
        media_download_headers: headers for downloading media
        context: a dictionary to allow custom vectors in search
        score_modifiers: a dictionary to modify the score based on field values, for tensor search only
        model_auth: Authorisation details for downloading a model (if required)
        highlights: if True, highlights will be returned
        text_query_prefix: prefix to add to text queries
        rerank_depth: the number of hits per shard during retrieval
        interpolation_method: the method to use for combining vectors
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
    RequestMetricsStore.for_request().start("search.vector.processing_before_vespa")

    index_name = marqo_index.name

    # Determine the text query prefix
    text_query_prefix = marqo_index.model.get_text_query_prefix(text_query_prefix)

    if isinstance(query, CustomVectorQuery):
        if context is None:
            context = SearchContext(
                tensor=[SearchContextTensor(vector=query.customVector.vector, weight=1)]
            )
        else:
            context.tensor.append(SearchContextTensor(vector=query.customVector.vector, weight=1))
        query = None

    queries = [BulkSearchQueryEntity(
        q=query, searchableAttributes=searchable_attributes, searchMethod=SearchMethod.TENSOR, limit=result_count,
        offset=offset, showHighlights=False, filter=filter_string, attributesToRetrieve=attributes_to_retrieve,
        boost=boost, mediaDownloadHeaders=media_download_headers, context=context, scoreModifiers=score_modifiers,
        index=marqo_index, modelAuth=model_auth, text_query_prefix=text_query_prefix, rerankDepth=rerank_depth
    )]

    with RequestMetricsStore.for_request().time(f"search.vector_inference_full_pipeline"):
        qidx_to_vectors: Dict[Qidx, List[float]] = run_vectorise_pipeline(config, queries, device, interpolation_method)
    vectorised_text = list(qidx_to_vectors.values())[0]

    if not vectorised_text: # pragma: no cover
        raise InternalError(f"No vector is generated for the tensor query: {query}. ")

    marqo_query = MarqoTensorQuery(
        index_name=index_name,
        vector_query=vectorised_text,
        filter=filter_string,
        limit=result_count,
        ef_search=ef_search,
        approximate=approximate,
        approximate_threshold=approximate_threshold,
        offset=offset,
        searchable_attributes=searchable_attributes,
        attributes_to_retrieve=attributes_to_retrieve,
        score_modifiers=score_modifiers.to_marqo_score_modifiers() if score_modifiers is not None else None,
        rerank_depth_tensor=rerank_depth
    )

    vespa_index = vespa_index_factory(marqo_index)
    vespa_query = vespa_index.to_vespa_query(marqo_query)

    total_preprocess_time = RequestMetricsStore.for_request().stop("search.vector.processing_before_vespa")
    logger.debug(
        f"search (tensor) pre-processing: took {(total_preprocess_time):.3f}ms to vectorize and process query.")

    # SEARCH TIMER-LOGGER (roundtrip)
    with RequestMetricsStore.for_request().time("search.vector.vespa",
                                                lambda t: logger.debug(f"Vespa search: took {t:.3f}ms")
                                                ):
        try:
            responses = config.vespa_client.query(**vespa_query)
        except VespaStatusError as e:
            # The index will not have the embedding_similarity rank profile if there are no tensor fields
            if f"No profile named '{RANK_PROFILE_EMBEDDING_SIMILARITY}'" in e.message:
                raise core_exceptions.InvalidArgumentError(
                    f"Index {index_name} has no tensor fields, thus tensor search cannot be performed. "
                    f"Please create an index with a tensor field, or try a different search method."
                )
            raise e

    if not approximate and (responses.root.coverage.coverage < 100 or responses.root.coverage.degraded is not None):
        raise errors.InternalError(
            f'Graceful degradation detected for non-approximate search. '
            f'Coverage is not 100%: {responses.root.coverage}'
            f'Degraded: {str(responses.root.coverage.degraded)}'
        )

    # SEARCH TIMER-LOGGER (post-processing)
    RequestMetricsStore.for_request().start("search.vector.postprocess")
    gathered_docs = gather_documents_from_response(responses, marqo_index, highlights, attributes_to_retrieve)

    if boost is not None:
        raise api_exceptions.MarqoWebError('Boosting is not currently supported with Vespa')

    total_postprocess_time = RequestMetricsStore.for_request().stop("search.vector.postprocess")
    logger.debug(
        f"search (tensor) post-processing: took {(total_postprocess_time):.3f}ms to sort and format "
        f"{len(gathered_docs)} results from Vespa."
    )

    return gathered_docs


def delete_index(config: Config, index_name):
    config.index_management.delete_index_by_name(index_name)
    if index_name in get_cache():
        del get_cache()[index_name]


# TODO [Refactoring device logic] move to device manager
def get_cpu_info() -> dict:
    return {
        "cpu_usage_percent": f"{psutil.cpu_percent(1)} %",  # The number 1 is a time interval for CPU usage calculation.
        "memory_used_percent": f"{psutil.virtual_memory()[2]} %",
        # The number 2 is just an index number to get the expected results
        "memory_used_gb": f"{round(psutil.virtual_memory()[3] / 1000000000, 1)}",
        # The number 3 is just an index number to get the expected results
    }


def delete_documents(config: Config, index_name: str, doc_ids: List[str]):
    """Delete documents from the Marqo index with the given doc_ids """
    # Make sure the index exists
    marqo_index = index_meta_cache.get_index(index_management=config.index_management, index_name=index_name)

    return delete_docs.delete_documents(
        config=config,
        del_request=MqDeleteDocsRequest(
            index_name=index_name,
            schema_name=marqo_index.schema_name,
            document_ids=doc_ids,
        )
    )


def get_embedding_field_names(marqo_index: MarqoIndex, tensor_field_names: Optional[List[str]] = None) \
        -> (Tuple)[List[str], List[str]]:
    """
    Get the Vespa field names for embeddings based on the index type.
    
    Args:
        marqo_index: The Marqo index object
        tensor_field_names: Specific tensor fields to get embeddings for. If None, get all.
    
    Returns:
        List of Marqo tensor field names and Vespa field names for embeddings
        marqo_field_names, vespa_field_names

        For structured/semistructured: returned marqo_field_names can never be None.
        For unstructured: returned marqo_field_names can be None if no tensor fields are specified.
    """
    
    if marqo_index.type in {IndexType.Structured, IndexType.SemiStructured}:
        # For structured indexes, embeddings are stored per field
        if hasattr(marqo_index, 'tensor_fields'):
            if tensor_field_names:
                index_tensor_field_names = [tf.name for tf in marqo_index.tensor_fields]
                requested_tensor_fields = []
                for tf_name in tensor_field_names:
                    if tf_name in index_tensor_field_names:
                        # If tf_name is in index_tensor_field_names, append the corresponding item
                        # in marqo_index.tensor_fields that has that name
                        requested_tensor_fields.append(
                            next(tf for tf in marqo_index.tensor_fields if tf.name == tf_name)
                        )
                    else:
                        raise core_exceptions.InvalidArgumentError(
                            f"Tensor field '{tf_name}' not found in index '{marqo_index.name}'. "
                            f"Available tensor fields: {index_tensor_field_names}"
                        )
            else:
                requested_tensor_fields = marqo_index.tensor_fields
            
            return ([tf.name for tf in requested_tensor_fields],
                    [tf.embeddings_field_name for tf in requested_tensor_fields])
        else:
            # Index has no tensor fields at all
            raise core_exceptions.InvalidArgumentError(
                f"Index '{marqo_index.name}' has no tensor fields, cannot retrieve embeddings"
                f" for {tensor_field_names}"
            )
    else:
        raise InternalError(
            f"Attempting to retrieve only embeddings for unstructured index '{marqo_index.name}'"
            f" which was created before {MINIMUM_SEMI_STRUCTURED_INDEX_VERSION}. This functionality should be disabled."
        )


def get_doc_vectors_per_tensor_field_by_ids(
    config: Config, 
    index_name: str, 
    document_ids: List[str],
    tensor_fields: Optional[List[str]] = None,
    allow_missing_documents: bool = False,
) -> Dict[str, Dict[str, List[List[float]]]]:
    """
    Get only the embeddings for documents by their IDs.
    
    Args:
        config: Marqo config
        index_name: Name of the index
        document_ids: List of document IDs to fetch
        tensor_fields: Specific tensor fields to get. If None, get all tensor fields.
        allow_missing_documents: If True, will not raise an error if a document is not found
    
    Returns:
        Dict mapping document_id to field_name to list of embedding vectors
        E.g.,
        {
            "doc_id_1": {
                "field_name_1": [[0.1, 0.2, ...], ...],
                "field_name_2": [[0.3, 0.4, ...], ...],
            },
            "doc_id_2": {"field_name_1": [[0.5, 0.6, ...], ...]}
         }
    """

    # We can just use the cache here since we refresh every 1s.
    marqo_index = index_meta_cache.get_index(index_management=config.index_management, index_name=index_name)
    
    # Get the embedding field names we want to retrieve
    viable_tensor_fields, embedding_fields = get_embedding_field_names(marqo_index, tensor_fields)
    
    # Add the document ID field so we can identify the documents (structured and unstructured are the same here)
    fields_to_retrieve = [structured_common.FIELD_ID] + embedding_fields
    
    # Get documents with only embedding fields
    with RequestMetricsStore.for_request().time(f"get_document_vectors.vespa"):
        batch_get = config.vespa_client.get_batch(
            document_ids,
            marqo_index.schema_name,
            fields=fields_to_retrieve
        )
    
    vespa_index = vespa_index_factory(marqo_index)
    result = {}

    # Using index so correct document_id can be fetched for error message if needed
    for res_idx in range(len(batch_get.responses)):
        response = batch_get.responses[res_idx]

        if response.status == 200:
            # Extract vectors directly (for structured and semi-structured)
            # Skip turning into marqo document
            raw_response_dict = response.document.fields
            doc_id = raw_response_dict["marqo__id"]

            # Initialize the result for this document ID
            result[doc_id] = {}
            for tf_idx in range(len(viable_tensor_fields)):
                # Get marqo tensor field name from vespa field name
                marqo_tensor_field_name = viable_tensor_fields[tf_idx]
                retrieved_embedding_field_name = embedding_fields[tf_idx]

                if retrieved_embedding_field_name in raw_response_dict:
                    try:
                        # If the field exists, add all the tensors to the result
                        result[doc_id][marqo_tensor_field_name] = list(raw_response_dict
                                                                       [retrieved_embedding_field_name]["blocks"].values())
                    except (KeyError, AttributeError, TypeError) as e:
                        raise core_exceptions.VespaDocumentParsingError(
                            f'Cannot parse Vespa doc embeddings field {retrieved_embedding_field_name} '
                            f'with value {raw_response_dict[retrieved_embedding_field_name]}'
                        ) from e
                else:
                    # Otherwise, field is empty list
                    result[doc_id][marqo_tensor_field_name] = []
        elif response.status == 404 and allow_missing_documents:
                # If the document is not found and we are allowing missing documents, continue to next response
                continue
        else:
            # If the response is not successful, error out
            raise core_exceptions.InvalidArgumentError(
                f"Failed to retrieve document {document_ids[res_idx]} from index {index_name}. "
                f"Response status: {response.status}, message: {response.message}"
            )
    return result


