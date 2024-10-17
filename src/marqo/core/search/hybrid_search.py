from typing import List, Optional, Union, Iterable, Dict

from marqo.api import exceptions as api_exceptions
from marqo.api import exceptions as errors
# We depend on _httprequests.py for now, but this may be replaced in the future, as
# _httprequests.py is designed for the client
from marqo.config import Config
from marqo.core import constants
from marqo.core import exceptions as core_exceptions
from marqo.core.models.hybrid_parameters import HybridParameters
from marqo.core.models.marqo_index import UnstructuredMarqoIndex, StructuredMarqoIndex, SemiStructuredMarqoIndex
from marqo.core.models.marqo_query import MarqoHybridQuery
from marqo.core.vespa_index.vespa_index import for_marqo_index as vespa_index_factory
from marqo.core.structured_vespa_index.common import RANK_PROFILE_HYBRID_CUSTOM_SEARCHER
from marqo.tensor_search import index_meta_cache
from marqo.tensor_search import utils
from marqo.tensor_search.enums import (
    SearchMethod
)
from marqo.tensor_search.models.api_models import BulkSearchQueryEntity, ScoreModifierLists, CustomVectorQuery
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.models.search import Qidx, SearchContext, SearchContextTensor
from marqo.tensor_search.telemetry import RequestMetricsStore
from marqo.tensor_search.tensor_search import run_vectorise_pipeline, gather_documents_from_response, logger
from marqo.vespa.exceptions import VespaStatusError
import semver


class HybridSearch:
    def search(
            self, config: Config, index_name: str, query: Optional[Union[str, CustomVectorQuery]],
            result_count: int = 5,
            offset: int = 0, ef_search: Optional[int] = None, approximate: bool = True,
            searchable_attributes: Iterable[str] = None, filter_string: str = None, device: str = None,
            attributes_to_retrieve: Optional[List[str]] = None, boost: Optional[Dict] = None,
            image_download_headers: Optional[Dict] = None, context: Optional[SearchContext] = None,
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
                searchable_attributes: Iterable of field names to search. Should be None for hybrid search, or will
                raise validation error in MarqoHybridQuery
                verbose: if 0 - nothing is printed. if 1 - data is printed without vectors, if 2 - full
                    objects are printed out
                attributes_to_retrieve: if set, only returns these fields
                image_download_headers: headers for downloading images
                context: a dictionary to allow custom vectors in search
                score_modifiers: a dictionary to modify the score based on field values, should be None for hybrid search
                model_auth: Authorisation details for downloading a model (if required)
                highlights: if True, highlights will be returned
                text_query_prefix: prefix for text queries (for vectorisation only)
                hybrid_parameters: HybridParameters object to specify all parameters for hybrid search. If not provided,
                    default values will be used.
            Returns:

            Output format:
                [
                    {
                        "_id": doc_id
                        "doc": {# original document},
                        "highlights":[{}],
                        "_score": score,
                        "_lexical_score": RRF raw lexical score (if any)
                        "_tensor_score": tensor score (if any)
                    },
                ]
            """

        # # SEARCH TIMER-LOGGER (pre-processing)
        if not device:
            raise api_exceptions.InternalError("_hybrid_search cannot be called without `device`!")
        if boost is not None:
            raise api_exceptions.MarqoWebError('Boosting is not currently supported with Vespa')

        RequestMetricsStore.for_request().start("search.hybrid.processing_before_vespa")

        marqo_index = index_meta_cache.get_index(index_management=config.index_management, index_name=index_name)

        # Version checks (different for structured and unstructured)
        marqo_index_version = marqo_index.parsed_marqo_version()
        if isinstance(marqo_index, StructuredMarqoIndex) and \
                marqo_index_version < constants.MARQO_STRUCTURED_HYBRID_SEARCH_MINIMUM_VERSION:
            raise core_exceptions.UnsupportedFeatureError(
                f"Hybrid search is only supported for Marqo structured indexes created with Marqo "
                f"{str(constants.MARQO_STRUCTURED_HYBRID_SEARCH_MINIMUM_VERSION)} or later. "
                f"This index was created with Marqo {marqo_index_version}."
            )
        elif isinstance(marqo_index, UnstructuredMarqoIndex) and \
                marqo_index_version < constants.MARQO_UNSTRUCTURED_HYBRID_SEARCH_MINIMUM_VERSION:
            raise core_exceptions.UnsupportedFeatureError(
                f"Hybrid search is only supported for Marqo unstructured indexes created with Marqo "
                f"{str(constants.MARQO_UNSTRUCTURED_HYBRID_SEARCH_MINIMUM_VERSION)} or later. "
                f"This index was created with Marqo {marqo_index_version}."
            )

        # Use default hybrid settings if not provided
        if hybrid_parameters is None:
            hybrid_parameters = HybridParameters()

        # TODO: Remove when unstructured searchable attributes are supported
        if (isinstance(marqo_index, UnstructuredMarqoIndex) and
                not isinstance(marqo_index, SemiStructuredMarqoIndex) and
                (hybrid_parameters.searchableAttributesTensor is not None or
                 hybrid_parameters.searchableAttributesLexical is not None)):
            raise core_exceptions.UnsupportedFeatureError(
                f"Hybrid search for unstructured indexes currently does not support `searchableAttributesTensor` or "
                f"`searchableAttributesLexical`. Please set these attributes to None."
            )

        # Determine the text query prefix
        text_query_prefix = marqo_index.model.get_text_query_prefix(text_query_prefix)

        # Edge cases for q data type
        if isinstance(query, CustomVectorQuery):
            query_text_vectorise = None
            query_text_search = query.customVector.content

            if context is None:
                context = SearchContext(
                    tensor=[SearchContextTensor(vector=query.customVector.vector, weight=1)]
                )
            else:
                context.tensor.append(SearchContextTensor(vector=query.customVector.vector, weight=1))
        elif query is None:
            # This is only acceptable if retrieval_method="tensor", ranking_method="tensor", and context exists.
            # Treated like normal tensor search with context.
            if not (hybrid_parameters.retrievalMethod.upper() == SearchMethod.TENSOR and
                    hybrid_parameters.rankingMethod.upper() == SearchMethod.TENSOR):
                raise core_exceptions.InvalidArgumentError(
                    "Query cannot be 'None' for hybrid search unless retrieval_method and ranking_method "
                    "are both 'tensor'.")
            if context is None:
                raise core_exceptions.InvalidArgumentError(
                    "Query cannot be 'None' for hybrid search unless 'context' is provided.")
            query_text_vectorise = None
            query_text_search = None

        else:  # string or dict query
            query_text_vectorise = query
            query_text_search = query

        queries = [BulkSearchQueryEntity(
            q=query_text_vectorise, searchableAttributes=searchable_attributes, searchMethod=SearchMethod.HYBRID,
            limit=result_count,
            offset=offset, showHighlights=False, filter=filter_string, attributesToRetrieve=attributes_to_retrieve,
            boost=boost, image_download_headers=image_download_headers, context=context, scoreModifiers=score_modifiers,
            index=marqo_index, modelAuth=model_auth, text_query_prefix=text_query_prefix,
            hybridParameters=hybrid_parameters
        )]

        with RequestMetricsStore.for_request().time(f"search.hybrid.vector_inference_full_pipeline"):
            qidx_to_vectors: Dict[Qidx, List[float]] = run_vectorise_pipeline(config, queries, device)
        vectorised_text = list(qidx_to_vectors.values())[0]

        # Parse text into required and optional terms.
        if query_text_search:
            (required_terms, optional_terms) = utils.parse_lexical_query(query_text_search)
        else:
            required_terms = []
            optional_terms = []

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
            score_modifiers_lexical=hybrid_parameters.scoreModifiersLexical.to_marqo_score_modifiers()
            if hybrid_parameters.scoreModifiersLexical is not None else None,
            score_modifiers_tensor=hybrid_parameters.scoreModifiersTensor.to_marqo_score_modifiers()
            if hybrid_parameters.scoreModifiersTensor is not None else None,
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
            try:
                responses = config.vespa_client.query(**vespa_query)
            except VespaStatusError as e:
                # The index will not have the embedding_similarity rank profile if there are no tensor fields
                if f"No profile named '{RANK_PROFILE_HYBRID_CUSTOM_SEARCHER}'" in e.message:
                    raise core_exceptions.InvalidArgumentError(
                        f"Index {index_name} either has no tensor fields or no lexically searchable fields, "
                        f"thus hybrid search cannot be performed. "
                        f"Please create an index with both tensor and lexical fields, or try a different search method."
                    )
                raise e

        if not approximate and (responses.root.coverage.coverage < 100 or responses.root.coverage.degraded is not None):
            raise errors.InternalError(
                f'Graceful degradation detected for non-approximate search. '
                f'Coverage is not 100%: {responses.root.coverage}'
                f'Degraded: {str(responses.root.coverage.degraded)}'
            )

        # SEARCH TIMER-LOGGER (post-processing)
        RequestMetricsStore.for_request().start("search.hybrid.postprocess")
        gathered_docs = gather_documents_from_response(responses, marqo_index, highlights, attributes_to_retrieve)

        total_postprocess_time = RequestMetricsStore.for_request().stop("search.hybrid.postprocess")
        logger.debug(
            f"search (hybrid) post-processing: took {(total_postprocess_time):.3f}ms to sort and format "
            f"{len(gathered_docs)} results from Vespa."
        )

        return gathered_docs
