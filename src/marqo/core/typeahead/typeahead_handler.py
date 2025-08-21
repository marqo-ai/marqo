import hashlib
from typing import List, Dict, Any, Optional

from marqo.core import exceptions as core_exceptions
from marqo.core.typeahead.text_normalization import normalize_text
from marqo.core.typeahead.typeahead_vespa_schema import TypeaheadVespaSchema
from marqo.vespa.vespa_client import VespaClient


class TypeaheadHandler:
    """Handler for typeahead functionality."""

    def __init__(self, vespa_client: VespaClient, index_name: str):
        self.vespa_client = vespa_client
        self.index_name = index_name
        self.schema_generator = TypeaheadVespaSchema(index_name)
        self.typeahead_schema_name = self.schema_generator._get_typeahead_schema_name(index_name)

    def get_suggestions(self, input_text: str, limit: int = 10,
                        fuzzy_edit_distance: int = 2, min_fuzzy_match_length: int = 3,
                        popularity_weight: Optional[float] = None, bm25_weight: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Get query suggestions for the given input.
        
        Args:
            input_text: Partial user search input
            limit: Maximum number of suggestions to return
            fuzzy_edit_distance: Maximum edit distance for fuzzy matching
            min_fuzzy_match_length: Minimum length to switch to fuzzy matching
            popularity_weight: Weight for popularity score in ranking (optional)
            bm25_weight: Weight for BM25 score in ranking (optional)
            
        Returns:
            List of suggestion dictionaries with query and relevance score
        """
        if not input_text or not input_text.strip():
            return []

        # Normalize the input
        normalized_input = normalize_text(input_text.strip())
        if not normalized_input:
            return []

        # Tokenize by whitespace
        tokens = normalized_input.split()
        if not tokens:
            return []

        # Build YQL query conditions for each token
        retrieval_terms = []
        ranking_terms = []
        for token in tokens:
            if len(token) < min_fuzzy_match_length:
                # Use exact matching for short tokens
                retrieval_terms.append(f"query_words contains '{token}'")
            else:
                # Use fuzzy matching for longer tokens
                retrieval_terms.append(
                    f"query_words contains "
                    f"({{maxEditDistance:{fuzzy_edit_distance}, prefix:true}}fuzzy(\"{token}\"))"
                )

            ranking_terms.append(f"query_index contains '{token}'")

        # Create single YQL query that ORs all token conditions
        yql_retrieval = " OR ".join(retrieval_terms)
        yql_ranking = " OR ".join(ranking_terms)
        yql = (f"SELECT * FROM {self.typeahead_schema_name} WHERE rank({yql_retrieval}, {yql_ranking})")

        search_params = {
            "yql": yql,
            "hits": limit,
            "ranking": "suggestions-rank-profile"
        }

        # Add query features if weights are provided
        query_features = {}
        if popularity_weight is not None:
            query_features["popularity_weight"] = popularity_weight
        if bm25_weight is not None:
            query_features["bm25_weight"] = bm25_weight
        
        if query_features:
            search_params["query_features"] = query_features

        try:
            response = self.vespa_client.query(schema=self.typeahead_schema_name, **search_params)
            hits = response.hits
            suggestions = []

            for hit in hits:
                fields = hit.fields or {}
                query = fields.get("query")
                relevance = hit.relevance

                if query:
                    suggestions.append({
                        "query": query,
                        "relevance": relevance
                    })

            return suggestions
        except core_exceptions.IndexNotFoundError:
            # If schema doesn't exist, return empty list
            return []
        except (core_exceptions.BackendCommunicationError, core_exceptions.VespaDocumentParsingError) as e:
            raise core_exceptions.BackendCommunicationError(f"Failed to get suggestions: {str(e)}")

    def index_queries(self, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Index queries for typeahead suggestions.
        
        Args:
            queries: List of dictionaries with 'query' and 'rank' fields
            
        Returns:
            Dictionary with indexing results
        """
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
            doc_id = hashlib.sha256(query.encode('utf-8')).hexdigest()
            normalized_query = normalize_text(query)
            tokenized_query = normalized_query.split()

            if not tokenized_query:
                errors.append(f"No tokens generated for query: {query}")
                continue

            from marqo.vespa.models.vespa_document import VespaDocument
            vespa_doc = VespaDocument(
                id=doc_id,
                fields={
                    "query_words": tokenized_query,
                    "query_index": normalized_query,
                    "query": query,
                    "popularity": float(popularity),
                }
            )

            # Index document in Vespa
            try:
                response = self.vespa_client.feed_document(
                    document=vespa_doc,
                    schema=self.typeahead_schema_name
                )
                # If no exception was raised, the document was successfully indexed
                indexed_count += 1
            except (core_exceptions.BackendCommunicationError, core_exceptions.VespaDocumentParsingError) as e:
                errors.append(f"Failed to index query '{query}': {str(e)}")

        return {"indexed": indexed_count, "errors": errors}

    def delete_all_queries(self) -> None:
        """
        Delete all queries from the typeahead index.
        
        Returns:
            True if successful
        """
        self.vespa_client.delete_all_docs(self.typeahead_schema_name)

    def delete_queries(self, queries: List[str]) -> Dict[str, Any]:
        """
        Delete specific queries from the typeahead index.
        
        Args:
            queries: List of query strings to delete
            
        Returns:
            Dictionary with deletion results
        """
        ids = [hashlib.sha256(q.strip().encode('utf-8')).hexdigest() for q in queries]

        self.vespa_client.delete_batch(ids, schema=self.typeahead_schema_name)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about indexed queries.
        
        Returns:
            Dictionary with stats including indexed query count
        """
        try:
            # Count total documents in typeahead schema
            search_params = {
                "yql": f"SELECT * FROM {self.typeahead_schema_name} WHERE true",
                "hits": 0,  # We only want the count
                "summary": "minimal"
            }

            response = self.vespa_client.query(schema=self.typeahead_schema_name, **search_params)
            # Access total_count property from QueryResult
            total_count = response.total_count or 0
            return {"indexedQueries": total_count}
        except core_exceptions.IndexNotFoundError:
            # If schema doesn't exist, return 0
            return {"indexedQueries": 0}
        except (core_exceptions.BackendCommunicationError, core_exceptions.VespaDocumentParsingError) as e:
            raise core_exceptions.BackendCommunicationError(f"Failed to get typeahead stats: {str(e)}")
