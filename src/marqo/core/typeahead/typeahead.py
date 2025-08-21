import hashlib
import time
from typing import List, Dict, Any, Optional

from marqo.core import exceptions as core_exceptions
from marqo.core.typeahead.text_normalization import normalize_text, generate_prefixes
from marqo.core.typeahead.typeahead_vespa_schema import TypeaheadVespaSchema
from marqo.core.typeahead.models import TypeaheadRequest, TypeaheadResponse, TypeaheadSuggestion
from marqo.tensor_search import index_meta_cache
from marqo.vespa.vespa_client import VespaClient


class Typeahead:
    """Handler for typeahead functionality."""

    def __init__(self, vespa_client: VespaClient, index_management):
        self.vespa_client = vespa_client
        self.index_management = index_management

    def get_suggestions(self, index_name: str, request: TypeaheadRequest) -> TypeaheadResponse:
        """
        Get query suggestions with timing and response model.
        
        Args:
            index_name: Name of the index to get suggestions for
            request: TypeaheadRequest containing all parameters
            
        Returns:
            TypeaheadResponse with suggestions and processing time
        """
        start_time = time.time()

        # Check if index exists
        marqo_index = index_meta_cache.get_index(index_management=self.index_management, index_name=index_name)

        # Set up schema information for this index
        schema_generator = TypeaheadVespaSchema(index_name)
        typeahead_schema_name = schema_generator._get_typeahead_schema_name(index_name)

        if not request.q or not request.q.strip():
            return TypeaheadResponse(suggestions=[], processing_time_ms=0)

        # Normalize the input
        normalized_input = normalize_text(request.q.strip())
        if not normalized_input:
            return TypeaheadResponse(suggestions=[], processing_time_ms=0)

        # Tokenize by whitespace
        tokens = normalized_input.split()
        if not tokens:
            return TypeaheadResponse(suggestions=[], processing_time_ms=0)

        # Build YQL query conditions for each token
        retrieval_terms = []
        ranking_terms = []
        for token in tokens:
            if len(token) < request.min_fuzzy_match_length:
                # Use exact prefix matching for short tokens
                retrieval_terms.append(
                    f"query_words contains ({{prefix:true}}\"{token}\")"
                )
            else:
                # Use fuzzy matching for longer tokens
                retrieval_terms.append(
                    f"query_words contains "
                    f"({{maxEditDistance:{request.fuzzy_edit_distance}, prefix:true}}fuzzy(\"{token}\"))"
                )

            ranking_terms.append(f"query_index contains \"{token}\"")

        # Create single YQL query that ORs all token conditions
        yql_retrieval = " OR ".join(retrieval_terms)
        yql_ranking = " OR ".join(ranking_terms)
        yql = (f"SELECT * FROM {typeahead_schema_name} WHERE rank({yql_retrieval}, {yql_ranking})")

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
            query_features["bm25_weight"] = request.bm25_weight

        if query_features:
            search_params["query_features"] = query_features

        try:
            response = self.vespa_client.query(schema=typeahead_schema_name, **search_params)
            hits = response.hits
            suggestions_data = []

            for hit in hits:
                fields = hit.fields or {}
                query = fields.get("query")
                relevance = hit.relevance

                if query:
                    suggestions_data.append({
                        "suggestion": query,
                        "_score": relevance
                    })

            processing_time_ms = int((time.time() - start_time) * 1000)

            suggestions = [
                TypeaheadSuggestion(suggestion=item["suggestion"], score=item["_score"])
                for item in suggestions_data
            ]

            return TypeaheadResponse(
                suggestions=suggestions,
                processing_time_ms=processing_time_ms
            )
        except core_exceptions.IndexNotFoundError:
            # If schema doesn't exist, return empty response
            processing_time_ms = int((time.time() - start_time) * 1000)
            return TypeaheadResponse(suggestions=[], processing_time_ms=processing_time_ms)
        except (core_exceptions.BackendCommunicationError, core_exceptions.VespaDocumentParsingError) as e:
            raise core_exceptions.BackendCommunicationError(f"Failed to get suggestions: {str(e)}")

    def index_queries(self, index_name: str, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Index queries for typeahead suggestions.
        
        Args:
            index_name: Name of the index to index queries for
            queries: List of dictionaries with 'query' and 'rank' fields
            
        Returns:
            Dictionary with indexing results
        """
        # Check if index exists
        marqo_index = index_meta_cache.get_index(index_management=self.index_management, index_name=index_name)

        # Set up schema information for this index
        schema_generator = TypeaheadVespaSchema(index_name)
        typeahead_schema_name = schema_generator._get_typeahead_schema_name(index_name)

        if not queries:
            return {"indexed": 0, "errors": []}

        indexed_count = 0
        errors = []

        for query_data in queries:
            query = query_data.get("query", "").strip()
            popularity = query_data.get("popularity", 0.0)

            if not query:
                errors.append(f"Empty query in: {query_data}")
                continue

            # Generate document ID using hash of query to avoid duplicates
            normalized_query = normalize_text(query)
            tokenized_query = normalized_query.split()
            query_prefixes = generate_prefixes(normalized_query)

            doc_id = hashlib.sha256(normalized_query.encode('utf-8')).hexdigest()

            if not tokenized_query:
                errors.append(f"No tokens generated for query: {query}")
                continue

            from marqo.vespa.models.vespa_document import VespaDocument
            vespa_doc = VespaDocument(
                id=doc_id,
                fields={
                    "query_words": tokenized_query,
                    "query_index": " ".join(query_prefixes),
                    "query": query,
                    "popularity": float(popularity),
                }
            )

            # Index document in Vespa
            try:
                response = self.vespa_client.feed_document(
                    document=vespa_doc,
                    schema=typeahead_schema_name
                )
                # If no exception was raised, the document was successfully indexed
                indexed_count += 1
            except (core_exceptions.BackendCommunicationError, core_exceptions.VespaDocumentParsingError) as e:
                errors.append(f"Failed to index query '{query}': {str(e)}")

        return {"indexed": indexed_count, "errors": errors}

    def delete_all_queries(self, index_name: str) -> None:
        """
        Delete all queries from the typeahead index.
        
        Args:
            index_name: Name of the index to delete queries from
        """
        # Check if index exists
        marqo_index = index_meta_cache.get_index(index_management=self.index_management, index_name=index_name)

        # Set up schema information for this index
        schema_generator = TypeaheadVespaSchema(index_name)
        typeahead_schema_name = schema_generator._get_typeahead_schema_name(index_name)

        self.vespa_client.delete_all_docs(typeahead_schema_name)

    def delete_queries(self, index_name: str, queries: List[str]) -> Dict[str, Any]:
        """
        Delete specific queries from the typeahead index.
        
        Args:
            index_name: Name of the index to delete queries from
            queries: List of query strings to delete
            
        Returns:
            Dictionary with deletion results
        """
        # Check if index exists
        marqo_index = index_meta_cache.get_index(index_management=self.index_management, index_name=index_name)

        # Set up schema information for this index
        schema_generator = TypeaheadVespaSchema(index_name)
        typeahead_schema_name = schema_generator._get_typeahead_schema_name(index_name)

        ids = [hashlib.sha256(normalize_text(q).encode('utf-8')).hexdigest() for q in queries]

        self.vespa_client.delete_batch(ids, schema=typeahead_schema_name)

    def get_stats(self, index_name: str) -> Dict[str, Any]:
        """
        Get statistics about indexed queries.
        
        Args:
            index_name: Name of the index to get stats for
        
        Returns:
            Dictionary with stats including indexed query count
        """
        # Check if index exists
        marqo_index = index_meta_cache.get_index(index_management=self.index_management, index_name=index_name)

        # Set up schema information for this index
        schema_generator = TypeaheadVespaSchema(index_name)
        typeahead_schema_name = schema_generator._get_typeahead_schema_name(index_name)

        try:
            # Count total documents in typeahead schema
            search_params = {
                "yql": f"SELECT * FROM {typeahead_schema_name} WHERE true",
                "hits": 0,  # We only want the count
                "summary": "minimal"
            }

            response = self.vespa_client.query(schema=typeahead_schema_name, **search_params)
            # Access total_count property from QueryResult
            total_count = response.total_count or 0
            return {"indexedQueries": total_count}
        except core_exceptions.IndexNotFoundError:
            # If schema doesn't exist, return 0
            return {"indexedQueries": 0}
        except (core_exceptions.BackendCommunicationError, core_exceptions.VespaDocumentParsingError) as e:
            raise core_exceptions.BackendCommunicationError(f"Failed to get typeahead stats: {str(e)}")
