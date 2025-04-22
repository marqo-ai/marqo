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
            (the uncached field won’t be searched)
    Vector search:
        - Searching an existing but uncached field will return no results (the
            uncached field won’t be searched)
        - Searching all fields will return a poor result (the uncached field
            won’t be searched)

"""
import typing
from collections import defaultdict
from timeit import default_timer as timer
from typing import List, Optional, Union, Iterable, Sequence, Dict, Any, Tuple, Set

import numpy as np
import psutil

import marqo.core.unstructured_vespa_index.common as unstructured_common
from marqo import marqo_docs
from marqo.api import exceptions as api_exceptions
from marqo.api import exceptions as errors
from marqo.config import Config
from marqo.core import constants
from marqo.core import exceptions as core_exceptions
from marqo.core.inference.api import Modality, TextPreprocessingConfig, ImagePreprocessingConfig, \
    AudioPreprocessingConfig, VideoPreprocessingConfig, InferenceError, Inference, InferenceRequest, ModelConfig, \
    ModelError, InferenceErrorModel
from marqo.core.inference.modality_utils import infer_modality
from marqo.core.models.facets_parameters import FacetsParameters
from marqo.core.models.hybrid_parameters import HybridParameters
from marqo.core.models.marqo_get_documents_by_id_response import (MarqoGetDocumentsByIdsResponse,
                                                                  MarqoGetDocumentsByIdsItem)
from marqo.core.models.marqo_index import IndexType
from marqo.core.models.marqo_index import MarqoIndex
from marqo.core.models.marqo_query import MarqoTensorQuery, MarqoLexicalQuery
from marqo.core.semi_structured_vespa_index.semi_structured_vespa_schema import SemiStructuredVespaSchema
from marqo.core.structured_vespa_index.common import RANK_PROFILE_BM25, RANK_PROFILE_EMBEDDING_SIMILARITY
from marqo.core.vespa_index.vespa_index import for_marqo_index as vespa_index_factory
from marqo.exceptions import InternalError
from marqo.s2_inference import errors as s2_inference_errors
from marqo.s2_inference import s2_inference
from marqo.s2_inference.reranking import rerank
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
from marqo.tensor_search.models.delete_docs_objects import MqDeleteDocsRequest
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.models.search import Qidx, JHash, SearchContext, VectorisedJobs, VectorisedJobPointer, \
    SearchContextTensor, QueryContentCollector, QueryContent
from marqo.tensor_search.telemetry import RequestMetricsStore
from marqo.logging import get_logger
from marqo.vespa.exceptions import VespaStatusError
from marqo.vespa.models import QueryResult

logger = get_logger(__name__)


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


def rerank_query(query: BulkSearchQueryEntity, result: Dict[str, Any], reranker: Union[str, Dict], device: str,
                 num_highlights: int):
    if query.searchableAttributes is None:
        raise api_exceptions.InvalidArgError(
            f"searchable_attributes cannot be None when re-ranking. Specify which fields to search and rerank over.")
    try:
        start_rerank_time = timer()
        rerank.rerank_search_results(search_result=result, query=query.q,
                                     model_name=reranker, device=device,
                                     searchable_attributes=query.searchableAttributes, num_highlights=num_highlights)
        logger.debug(
            f"search ({query.searchMethod.lower()}) reranking using {reranker}: took {(timer() - start_rerank_time):.3f}s to rerank results.")
    except Exception as e:
        raise api_exceptions.BadRequestError(f"reranking failure due to {str(e)}")


def search(config: Config, index_name: str, text: Optional[Union[str, dict, CustomVectorQuery]],
           result_count: int = 3, offset: int = 0, rerank_depth: Optional[int] = None,
           highlights: bool = True, ef_search: Optional[int] = None,
           approximate: Optional[bool] = None,
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
        if not isinstance(attributes_to_retrieve, (List, typing.Tuple)):
            raise api_exceptions.InvalidArgError("attributes_to_retrieve must be a sequence!")
        [validation.validate_field_name(attribute) for attribute in attributes_to_retrieve]
    if verbose:
        print(f"determined_search_method: {search_method}, text query: {text}")

    selected_device = device

    # Fetch marqo index to pass to search method
    marqo_index = index_meta_cache.get_index(index_management=config.index_management, index_name=index_name)
    marqo_index_version = marqo_index.parsed_marqo_version()
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

        if search_method.upper() == SearchMethod.TENSOR:
            search_result = _vector_text_search(
                config=config, marqo_index=marqo_index, query=text, result_count=result_count, offset=offset,
                ef_search=ef_search, approximate=approximate, searchable_attributes=searchable_attributes,
                filter_string=filter, device=selected_device, attributes_to_retrieve=attributes_to_retrieve,
                boost=boost,
                media_download_headers=media_download_headers, context=context, score_modifiers=score_modifiers,
                model_auth=model_auth, highlights=highlights, text_query_prefix=text_query_prefix, rerank_depth=rerank_depth
            )
        elif search_method.upper() == SearchMethod.HYBRID:
            # TODO: Deal with circular import when all modules are refactored out.
            from marqo.core.search.hybrid_search import HybridSearch
            search_result = HybridSearch().search(
                config=config, marqo_index=marqo_index, query=text, result_count=result_count, offset=offset,
                rerank_depth=rerank_depth,
                ef_search=ef_search, approximate=approximate, searchable_attributes=searchable_attributes,
                filter_string=filter, device=selected_device, attributes_to_retrieve=attributes_to_retrieve,
                boost=boost,
                media_download_headers=media_download_headers, context=context, score_modifiers=score_modifiers,
                model_auth=model_auth, highlights=highlights, text_query_prefix=text_query_prefix,
                hybrid_parameters=hybrid_parameters, facets=facets, track_total_hits=track_total_hits
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
            score_modifiers=score_modifiers
        )
    else:
        raise api_exceptions.InvalidArgError(f"Search called with unknown search method: {search_method}")

    if reranker is not None:
        raise api_exceptions.InvalidArgError(f"Reranker is no longer supported in Marqo version 2.17 and later")

    if isinstance(text, CustomVectorQuery):
        search_result["query"] = text.dict()    # Make object JSON serializable
    else:
        search_result["query"] = text

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
        score_modifiers: Optional[ScoreModifierLists] = None):
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
        score_modifiers=score_modifiers.to_marqo_score_modifiers() if score_modifiers else None
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
    return QueryContentCollector(queries = query_content_list)


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
        if doc.id.startswith("group:facet:"): # Not an actual document id but group's id returned by vespa
            continue
        marqo_doc = vespa_index.to_marqo_document(dict(doc), return_highlights=highlights)
        marqo_doc['_score'] = doc.relevance

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
    if modality == Modality.TEXT:
        return TextPreprocessingConfig()   # the prefix has been added to the query, so we don't need to specify it here
    elif modality == Modality.IMAGE:
        return ImagePreprocessingConfig(download_header=media_download_headers, download_thread_count=1)
    elif modality == Modality.AUDIO:
        return AudioPreprocessingConfig(download_header=media_download_headers, download_thread_count=1)
    elif modality == Modality.VIDEO:
        return VideoPreprocessingConfig(download_header=media_download_headers, download_thread_count=1)
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
                model_config=ModelConfig(
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
        jobs: Dict[JHash, VectorisedJobs]
) -> Dict[Qidx, List[float]]:
    """
    Retrieve the vectorised content associated to each query from the set of batch vectorise jobs.
    Handles multi-modal queries, by weighting and combining queries into a single vector

    Args:
        - queries: Original search queries.
        - qidx_to_job: VectorisedJobPointer for each query
        - job_to_vectors: inference output from each VectorisedJob
        - config: standard Marqo config.

    """
    result: Dict[Qidx, List[float]] = defaultdict(list)
    for qidx, ptrs in qidx_to_job.items():

        # vectors = job_to_vectors[ptrs.job_hash][ptrs.start_idx: ptrs.end_idx]

        # qidx_to_vectors[qidx].append(vectors)
        q = queries[qidx]

        if isinstance(q.q, dict) or q.q is None:
            ordered_queries = list(q.q.items()) if isinstance(q.q, dict) else None
            weighted_vectors = []
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
                weighted_vectors = [np.asarray(vec) * weight for vec, weight, content in vectorised_ordered_queries]

            context_tensors = q.get_context_tensor()
            if context_tensors is not None:
                weighted_vectors += [np.asarray(v.vector) * v.weight for v in context_tensors]

            for vector in weighted_vectors:
                if not q.index.model.get_dimension() == len(vector):
                    raise api_exceptions.InvalidArgError(
                        f"The dimension of the vectors returned by the model or given by the context vectors "
                        f"does not match the expected dimension. "
                        f"Expected dimension {q.index.model.get_dimension()} but got {len(vector)}"
                    )

            merged_vector = np.mean(weighted_vectors, axis=0)

            if q.index.normalize_embeddings:
                norm = np.linalg.norm(merged_vector, axis=-1, keepdims=True)
                if norm > 0:
                    merged_vector /= np.linalg.norm(merged_vector, axis=-1, keepdims=True)
            result[qidx] = list(merged_vector)
        elif isinstance(q.q, str):
            # result[qidx] = vectors[0]
            result[qidx] = get_content_vector(
                possible_jobs=qidx_to_job.get(qidx, []),
                job_to_vectors=job_to_vectors,
                content=q.q
            )
        else:
            raise ValueError(f"Unexpected query type: {type(q.q).__name__}")
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


def run_vectorise_pipeline(config: Config, queries: List[BulkSearchQueryEntity], device: Union[Device, str]) -> Dict[
    Qidx, List[float]]:
    """Run the query vectorisation process

    Raise:
        api_exceptions.InvalidArgError: If the vectorisation process fails or if the media cannot be downloaded.
    """

    # Prepend the prefixes to the queries if it exists (output should be of type List[BulkSearchQueryEntity])
    try:
        prefixed_queries = add_prefix_to_queries(queries)
    except s2_inference_errors.MediaDownloadError as e:
        raise api_exceptions.InvalidArgError(message=str(e)) from e

    # 1. Pre-process inputs ready for s2_inference.vectorise
    # we can still use qidx_to_job. But the jobs structure may need to be different
    vector_jobs_tuple: Tuple[Dict[Qidx, List[VectorisedJobPointer]], Dict[JHash, VectorisedJobs]] = create_vector_jobs(
        prefixed_queries, config, device)

    qidx_to_jobs, jobs = vector_jobs_tuple

    # 2. Vectorise in batches against all queries
    ## TODO: To ensure that we are vectorising in batches, we can mock vectorise (), and see if the number of calls is as expected (if batch_size = 16, and number of docs = 32, and all args are the same, then number of calls = 2)
    # TODO: we need to enable str/PIL image structure:
    job_ptr_to_vectors: Dict[JHash, Dict[str, List[float]]] = vectorise_jobs(config.inference, list(jobs.values()))

    # 3. For each query, get associated vectors
    qidx_to_vectors: Dict[Qidx, List[float]] = get_query_vectors_from_jobs(
        prefixed_queries, qidx_to_jobs, job_ptr_to_vectors, config, jobs
    )
    return qidx_to_vectors


def _vector_text_search(
        config: Config, marqo_index: MarqoIndex,
        query: Optional[Union[str, dict, CustomVectorQuery]], result_count: int = 5,
        offset: int = 0,
        ef_search: Optional[int] = None, approximate: bool = True,
        searchable_attributes: Iterable[str] = None, filter_string: str = None, device: str = None,
        attributes_to_retrieve: Optional[List[str]] = None, boost: Optional[Dict] = None,
        media_download_headers: Optional[Dict] = None, context: Optional[SearchContext] = None,
        score_modifiers: Optional[ScoreModifierLists] = None, model_auth: Optional[ModelAuth] = None,
        highlights: bool = False, text_query_prefix: Optional[str] = None, rerank_depth: Optional[int] = None
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
        qidx_to_vectors: Dict[Qidx, List[float]] = run_vectorise_pipeline(config, queries, device)
    vectorised_text = list(qidx_to_vectors.values())[0]

    marqo_query = MarqoTensorQuery(
        index_name=index_name,
        vector_query=vectorised_text,
        filter=filter_string,
        limit=result_count,
        ef_search=ef_search,
        approximate=approximate,
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


def get_loaded_models() -> dict:
    available_models = s2_inference.get_available_models()
    message = {"models": []}

    for ix in available_models:
        if isinstance(ix, str):
            message["models"].append({"model_name": ix.split("||")[0], "model_device": ix.split("||")[-1]})
    return message


def eject_model(model_name: str, device: str) -> dict:
    try:
        result = s2_inference.eject_model(model_name, device)
    except s2_inference_errors.ModelNotInCacheError as e:
        raise api_exceptions.ModelNotInCacheError(message=str(e))
    return result


# TODO [Refactoring device logic] move to device manager
def get_cpu_info() -> dict:
    return {
        "cpu_usage_percent": f"{psutil.cpu_percent(1)} %",  # The number 1 is a time interval for CPU usage calculation.
        "memory_used_percent": f"{psutil.virtual_memory()[2]} %",
        # The number 2 is just a index number to get the expected results
        "memory_used_gb": f"{round(psutil.virtual_memory()[3] / 1000000000, 1)}",
        # The number 3 is just a index number to get the expected results
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


