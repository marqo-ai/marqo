import json
import time
import uuid
from typing import List, Dict, Any, Optional

from marqo.vespa.vespa_client import VespaClient
from marqo.core.typeahead.text_normalization import normalize_text, generate_suffixes, calculate_edit_distance
from marqo.core.typeahead.typeahead_vespa_schema import TypeaheadVespaSchema
from marqo.core import exceptions as core_exceptions


class TypeaheadHandler:
    """Handler for typeahead functionality."""
    
    def __init__(self, vespa_client: VespaClient, index_name: str):
        self.vespa_client = vespa_client
        self.index_name = index_name
        self.schema_generator = TypeaheadVespaSchema(index_name)
        self.typeahead_schema_name = self.schema_generator._get_typeahead_schema_name(index_name)
    
    def get_suggestions(self, input_text: str, max_suggestions: int = 10, 
                       fuzzy_edit_distance: int = 2, min_fuzzy_match_length: int = 3) -> List[Dict[str, Any]]:
        """
        Get query suggestions for the given input.
        
        Args:
            input_text: Partial user search input
            max_suggestions: Maximum number of suggestions to return
            fuzzy_edit_distance: Maximum edit distance for fuzzy matching
            min_fuzzy_match_length: Minimum length to switch to fuzzy matching
            
        Returns:
            List of suggestion dictionaries with query and relevance score
        """
        if not input_text or not input_text.strip():
            return []
        
        normalized_input = normalize_text(input_text.strip())
        if not normalized_input:
            return []
        
        if len(normalized_input) < min_fuzzy_match_length:
            # Use exact prefix matching
            suggestions = self._get_exact_suggestions(normalized_input, max_suggestions)
        else:
            # Use fuzzy matching
            suggestions = self._get_fuzzy_suggestions(normalized_input, max_suggestions, fuzzy_edit_distance)
        
        return suggestions
    
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
        
        try:
            for query_data in queries:
                query = query_data.get("query", "").strip()
                rank = query_data.get("rank", 0.0)
                
                if not query:
                    errors.append(f"Empty query in: {query_data}")
                    continue
                
                # Generate document for Vespa
                doc_id = str(uuid.uuid4())
                suffixes = generate_suffixes(query)
                
                if not suffixes:
                    errors.append(f"No suffixes generated for query: {query}")
                    continue
                
                from marqo.vespa.models.vespa_document import VespaDocument
                vespa_doc = VespaDocument(
                    id=doc_id,
                    fields={
                        "query_suffixes": suffixes,
                        "original_query": query,
                        "rank": float(rank),
                        "query_id": doc_id
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
        except core_exceptions.MarqoError:
            # Re-raise known Marqo errors
            raise
    
    def delete_all_queries(self) -> bool:
        """
        Delete all queries from the typeahead index.
        
        Returns:
            True if successful
        """
        try:
            # Query all documents and delete them
            search_params = {
                "yql": f"SELECT * FROM {self.typeahead_schema_name} WHERE true",
                "hits": 1000  # Batch size for deletion
            }
            
            while True:
                try:
                    search_response = self.vespa_client.query(schema=self.typeahead_schema_name, **search_params)
                    hits = search_response.hits
                    
                    if not hits:
                        break
                    
                    # Delete documents in this batch
                    for hit in hits:
                        doc_id = hit.id.split("::")[-1] if hit.id else None  # Extract doc ID from Vespa ID format
                        if doc_id:
                            self.vespa_client.delete_document(id=doc_id, schema=self.typeahead_schema_name)
                    
                    # If we got fewer hits than requested, we're done
                    if len(hits) < search_params["hits"]:
                        break
                except (core_exceptions.BackendCommunicationError, core_exceptions.IndexNotFoundError) as e:
                    # If query fails due to communication or missing index, stop deletion
                    raise core_exceptions.InternalError(f"Failed to query documents for deletion: {str(e)}")
            
            return True
        except core_exceptions.MarqoError:
            # Re-raise known Marqo errors
            raise
    
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
    
    def _get_exact_suggestions(self, normalized_input: str, max_suggestions: int) -> List[Dict[str, Any]]:
        """Get suggestions using exact prefix matching."""
        search_params = {
            "yql": f"SELECT * FROM {self.typeahead_schema_name} WHERE query_suffixes contains '{normalized_input}'",
            "hits": max_suggestions,
            "ranking": "default"
        }
        
        try:
            response = self.vespa_client.query(schema=self.typeahead_schema_name, **search_params)
            hits = response.hits  # Use the hits property from QueryResult
            suggestions = []
            
            for hit in hits:
                fields = hit.fields or {}
                original_query = fields.get("original_query")
                relevance = hit.relevance
                
                if original_query:
                    suggestions.append({
                        "query": original_query,
                        "relevance": relevance
                    })
            
            return suggestions
        except core_exceptions.IndexNotFoundError:
            # If schema doesn't exist, return empty list
            return []
        except (core_exceptions.BackendCommunicationError, core_exceptions.VespaDocumentParsingError) as e:
            raise core_exceptions.BackendCommunicationError(f"Failed to get exact suggestions: {str(e)}")
    
    def _get_fuzzy_suggestions(self, normalized_input: str, max_suggestions: int, max_edit_distance: int) -> List[Dict[str, Any]]:
        """Get suggestions using fuzzy matching."""
        # First try exact matching
        exact_suggestions = self._get_exact_suggestions(normalized_input, max_suggestions)
        
        if len(exact_suggestions) >= max_suggestions:
            return exact_suggestions[:max_suggestions]
        
        # Get more candidates for fuzzy matching
        search_params = {
            "yql": f"SELECT * FROM {self.typeahead_schema_name} WHERE true",
            "hits": max_suggestions * 3,  # Get more candidates for filtering
            "ranking": "fuzzy"
        }
        
        try:
            response = self.vespa_client.query(schema=self.typeahead_schema_name, **search_params)
            hits = response.hits
            fuzzy_suggestions = []
            exact_queries = {s["query"] for s in exact_suggestions}
            
            for hit in hits:
                fields = hit.fields or {}
                original_query = fields.get("original_query")
                
                if not original_query or original_query in exact_queries:
                    continue
                
                # Check if query matches fuzzy criteria
                normalized_query = normalize_text(original_query)
                edit_distance = calculate_edit_distance(normalized_input, normalized_query[:len(normalized_input)])
                
                if edit_distance <= max_edit_distance:
                    relevance = hit.relevance * (1.0 - edit_distance / (max_edit_distance + 1))
                    fuzzy_suggestions.append({
                        "query": original_query,
                        "relevance": relevance
                    })
        except core_exceptions.IndexNotFoundError:
            # If schema doesn't exist, return exact suggestions only
            return exact_suggestions
        except (core_exceptions.BackendCommunicationError, core_exceptions.VespaDocumentParsingError) as e:
            raise core_exceptions.BackendCommunicationError(f"Failed to get fuzzy suggestions: {str(e)}")
        
        # Combine and sort suggestions
        all_suggestions = exact_suggestions + fuzzy_suggestions
        all_suggestions.sort(key=lambda x: x["relevance"], reverse=True)
        
        return all_suggestions[:max_suggestions]