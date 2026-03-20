import time
from timeit import default_timer as timer
from typing import List, Dict, Any

import blake3

from marqo.core.constants import MARQO_TYPEAHEAD_SCHEMA_MINIMUM_VERSION, CHARACTERS_TO_BE_ESCAPED_IN_VESPA
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models.typeahead import (
    TypeaheadRequest, TypeaheadResponse, TypeaheadSuggestion,
    TypeaheadIndexingResponse, TypeaheadIndexingError, TypeaheadIndexingRequest,
    TypeaheadStatsResponse, TypeaheadQuery, TypeaheadGetQueriesResponse
)
from marqo.core.typeahead.text_normalization import normalize_text, generate_prefixes
from marqo.logging import get_logger
from marqo.tensor_search.utils import check_feature_support
from marqo.vespa.models.vespa_document import VespaDocument
from marqo.vespa.vespa_client import VespaClient

logger = get_logger(__name__)


class Typeahead:
    """Handler for typeahead functionality."""
    check_typeahead_support = check_feature_support(MARQO_TYPEAHEAD_SCHEMA_MINIMUM_VERSION, 'Typeahead')

    def __init__(self, vespa_client: VespaClient, index_management: IndexManagement):
        self.vespa_client = vespa_client
        self.index_management = index_management

    @check_typeahead_support
    def get_suggestions(self, index_name: str, request: TypeaheadRequest) -> TypeaheadResponse:
        """
        Get query suggestions with timing and response model.
        
        Args:
            index_name: Name of the index to get suggestions for
            request: TypeaheadRequest containing all parameters
            
        Returns:
            TypeaheadResponse with suggestions and processing time
        """
        start_time = timer()

        # Check if index exists and get typeahead schema name
        from marqo.tensor_search import index_meta_cache
        marqo_index = index_meta_cache.get_index(index_management=self.index_management, index_name=index_name)
        typeahead_schema_name = marqo_index.typeahead_schema_name
        query = request.q.strip()

        if query == "":
            yql = f"SELECT query, metadata FROM {typeahead_schema_name} WHERE true"

        else:
            # Normalize the input
            normalized_input = normalize_text(query)
            if not normalized_input:
                return TypeaheadResponse(suggestions=[])

            # Tokenize by whitespace
            tokens = normalized_input.split()
            if not tokens:
                return TypeaheadResponse(suggestions=[])

            # Build YQL query conditions for each token
            retrieval_terms = []
            ranking_terms = []
            for token in tokens:
                escaped_token = self._escape_token(token)
                if len(token) < request.min_fuzzy_match_length:
                    # Use exact prefix matching for short tokens
                    retrieval_terms.append(
                        f"query_words contains ({{prefix:true}}\"{escaped_token}\")"
                    )
                else:
                    # Use fuzzy matching for longer tokens
                    retrieval_terms.append(
                        f"query_words contains "
                        f"({{maxEditDistance:{request.fuzzy_edit_distance}, prefix:true}}fuzzy(\"{escaped_token}\"))"
                    )

                ranking_terms.append(f"query_index contains \"{escaped_token}\"")

            # Join retrieval terms: AND when match_all_tokens mode, OR otherwise
            join_operator = " AND " if request.match_all_tokens else " OR "
            yql_retrieval = join_operator.join(retrieval_terms)
            yql_ranking = " OR ".join(ranking_terms)
            yql = f"SELECT query, metadata FROM {typeahead_schema_name} WHERE rank({yql_retrieval}, {yql_ranking})"

        search_params = {
            "yql": yql,
            "hits": request.limit,
            "ranking": "suggestions-rank-profile"
        }

        # Add query features if weights are provided
        query_features = {}
        if request.popularity_weight is not None:
            query_features["popularity_weight"] = request.popularity_weight
        if request.bm25_weight is not None:
            # bm25 wights is used to boost exact match result
            query_features["bm25_weight"] = request.bm25_weight

        if query_features:
            search_params["query_features"] = query_features

        response = self.vespa_client.query(schema=typeahead_schema_name, **search_params)
        hits = response.hits
        suggestions = []

        for hit in hits:
            fields = hit.fields
            query = fields["query"]
            metadata = fields.get("metadata", {})
            relevance = hit.relevance

            suggestions.append(
                TypeaheadSuggestion(suggestion=query, score=relevance, metadata=metadata)
            )

        processing_time_ms = round((timer() - start_time) * 1000)

        return TypeaheadResponse(
            suggestions=suggestions,
            processing_time_ms=processing_time_ms
        )

    @check_typeahead_support
    def index_queries(self, index_name: str, request: TypeaheadIndexingRequest) -> TypeaheadIndexingResponse:
        """
        Index queries for typeahead suggestions.
        
        Args:
            index_name: Name of the index to index queries for
            request: TypeaheadIndexRequest containing the queries
            
        Returns:
            TypeaheadIndexResponse with indexing results
        """
        start_time = timer()
        # Check if index exists and get typeahead schema name
        marqo_index = self.index_management.get_index(index_name=index_name)
        typeahead_schema_name = marqo_index.typeahead_schema_name

        if not request.queries:
            processing_time_ms = round((timer() - start_time) * 1000)
            return TypeaheadIndexingResponse(
                indexed=0,
                errors=[],
                processing_time_ms=processing_time_ms
            )

        indexed_count = 0
        errors = []
        vespa_docs = []
        normalised_query_map = {}  # map of normalised query and the original query, used for deduping
        doc_id_query_map = {}  # map of hash doc_id and the query, used for vespa response handling

        for add_query_request in request.queries:
            query = add_query_request.query
            normalized_query = normalize_text(query)

            if normalized_query in normalised_query_map:
                errors.append(TypeaheadIndexingError(
                    query=query,
                    message=f"Query is duplicate of {normalised_query_map[normalized_query]} "
                            f"after normalisation, will ignore",
                    code=400
                ))
                continue
            else:
                normalised_query_map[normalized_query] = query

            tokenized_query = normalized_query.split()
            query_prefixes = generate_prefixes(normalized_query)

            # Generate document ID using hash of query to avoid duplicates and special characters
            doc_id = self._generate_query_hash(normalized_query)
            doc_id_query_map[doc_id] = query

            if not tokenized_query:
                errors.append(TypeaheadIndexingError(query=query, message="No tokens generated for query", code=400))
                continue

            vespa_doc = VespaDocument(
                id=doc_id,
                fields={
                    "query_words": tokenized_query,
                    "query_index": " ".join(query_prefixes),
                    "query": query,
                    "popularity": add_query_request.popularity,
                    "metadata": add_query_request.metadata,
                    "last_updated_at": int(time.time())
                }
            )

            logger.debug("Adding typeahead vespa doc", vespa_doc)

            vespa_docs.append(vespa_doc)

        if vespa_docs:
            response = self.vespa_client.feed_batch(vespa_docs, schema=typeahead_schema_name)
            for resp in response.responses:
                doc_id = resp.id.split('::')[-1] if resp.id else None
                query = doc_id_query_map.get(doc_id, None)
                status, message = self.vespa_client.translate_vespa_document_response(resp.status, message=resp.message)
                if status != 200:
                    errors.append(TypeaheadIndexingError(query=query, message=message, code=status))
                else:
                    indexed_count += 1

        processing_time_ms = round((timer() - start_time) * 1000)
        return TypeaheadIndexingResponse(
            indexed=indexed_count, 
            errors=errors, 
            processing_time_ms=processing_time_ms
        )

    @check_typeahead_support
    def delete_all_queries(self, index_name: str) -> None:
        """
        Delete all queries from the typeahead index.
        
        Args:
            index_name: Name of the index to delete queries from
        """
        # Check if index exists and get typeahead schema name
        marqo_index = self.index_management.get_index(index_name=index_name)
        typeahead_schema_name = marqo_index.typeahead_schema_name

        self.vespa_client.delete_all_docs(typeahead_schema_name)

    @check_typeahead_support
    def delete_queries(self, index_name: str, queries: List[str]) -> Dict[str, Any]:
        """
        Delete specific queries from the typeahead index.
        
        Args:
            index_name: Name of the index to delete queries from
            queries: List of query strings to delete
            
        Returns:
            Dictionary with deletion results
        """
        # Check if index exists and get typeahead schema name
        marqo_index = self.index_management.get_index(index_name=index_name)
        typeahead_schema_name = marqo_index.typeahead_schema_name

        ids = [self._generate_query_hash(normalize_text(q)) for q in queries]

        # TODO process DeleteBatchResponse and return an appropriate API response
        self.vespa_client.delete_batch(ids, schema=typeahead_schema_name)

    @check_typeahead_support
    def get_stats(self, index_name: str) -> TypeaheadStatsResponse:
        """
        Get statistics about indexed queries.
        
        Args:
            index_name: Name of the index to get stats for
        
        Returns:
            TypeaheadStatsResponse with stats including indexed query count
        """
        # Check if index exists and get typeahead schema name
        marqo_index = self.index_management.get_index(index_name=index_name)
        typeahead_schema_name = marqo_index.typeahead_schema_name

        # Count total documents in typeahead schema
        search_params = {
            "yql": f"SELECT * FROM {typeahead_schema_name} WHERE true",
            "hits": 0,  # We only want the count
            "summary": "minimal"
        }

        response = self.vespa_client.query(schema=typeahead_schema_name, **search_params)
        # Access total_count property from QueryResult
        total_count = response.total_count or 0
        return TypeaheadStatsResponse(indexed_queries=total_count)

    @check_typeahead_support
    def get_queries(self, index_name: str, queries: List[str]) -> TypeaheadGetQueriesResponse:
        """
        Get queries from the typeahead index by query strings.
        
        Args:
            index_name: Name of the index to get queries from
            queries: List of query strings to retrieve
            
        Returns:
            TypeaheadGetQueriesResponse with matching queries
        """
        # Check if index exists and get typeahead schema name
        marqo_index = self.index_management.get_index(index_name=index_name)
        typeahead_schema_name = marqo_index.typeahead_schema_name
        
        if not queries:
            return TypeaheadGetQueriesResponse(queries=[])
        
        # Generate document IDs from normalized queries
        ids = [self._generate_query_hash(normalize_text(q)) for q in queries]
        
        # Get documents from Vespa
        response = self.vespa_client.get_batch(ids, schema=typeahead_schema_name)
        
        query_results = []
        for doc_response in response.responses:
            if doc_response.document and doc_response.document.fields:  # Document found
                fields = doc_response.document.fields
                query_results.append(TypeaheadQuery(**fields))
        
        return TypeaheadGetQueriesResponse(queries=query_results)

    def _escape_token(self, token: str) -> str:
        """Escape special characters in a token for Vespa YQL queries.

        Args:
            token: The token to escape

        Returns:
            The escaped token
        """
        escaped = []
        for char in token:
            if char in CHARACTERS_TO_BE_ESCAPED_IN_VESPA:
                escaped.append('\\' + char)
            else:
                escaped.append(char)
        return ''.join(escaped)

    def _generate_query_hash(self, query: str) -> str:
        """Generate a 128-bit blake3 hash for a query string.

        Args:
            query: The query string to hash

        Returns:
            32-character hexadecimal hash (128 bits)
        """
        return blake3.blake3(query.encode('utf-8')).digest(16).hex()
