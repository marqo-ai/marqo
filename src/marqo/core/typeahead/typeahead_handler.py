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
        
        try:
            if len(normalized_input) < min_fuzzy_match_length:
                # Use exact prefix matching
                suggestions = self._get_exact_suggestions(normalized_input, max_suggestions)
            else:
                # Use fuzzy matching
                suggestions = self._get_fuzzy_suggestions(normalized_input, max_suggestions, fuzzy_edit_distance)
            
            return suggestions
        except Exception as e:
            raise core_exceptions.InternalError(f"Error getting suggestions: {str(e)}")
    
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
                
                vespa_doc = {
                    "fields": {
                        "query_suffixes": suffixes,
                        "original_query": query,
                        "rank": float(rank),
                        "query_id": doc_id
                    }
                }
                
                # Index document in Vespa
                response = self.vespa_client.feed_document(
                    schema=self.typeahead_schema_name,
                    doc_id=doc_id,
                    document=vespa_doc
                )
                
                if response and response.status_code == 200:
                    indexed_count += 1
                else:
                    errors.append(f"Failed to index query: {query}")
            
            return {"indexed": indexed_count, "errors": errors}
        except Exception as e:
            raise core_exceptions.InternalError(f"Error indexing queries: {str(e)}")
    
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
                search_response = self.vespa_client.query(schema=self.typeahead_schema_name, **search_params)
                
                if not search_response or search_response.status_code != 200:
                    break
                
                hits = search_response.json().get("root", {}).get("children", [])
                if not hits:
                    break
                
                # Delete documents in this batch
                for hit in hits:
                    doc_id = hit.get("id", "").split("::")[-1]  # Extract doc ID from Vespa ID format
                    if doc_id:
                        self.vespa_client.delete_document(schema=self.typeahead_schema_name, doc_id=doc_id)
                
                # If we got fewer hits than requested, we're done
                if len(hits) < search_params["hits"]:
                    break
            
            return True
        except Exception as e:
            raise core_exceptions.InternalError(f"Error deleting queries: {str(e)}")
    
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
            
            if response and response.status_code == 200:
                total_count = response.json().get("root", {}).get("fields", {}).get("totalCount", 0)
                return {"indexedQueries": total_count}
            else:
                return {"indexedQueries": 0}
        except Exception as e:
            # If schema doesn't exist or other error, return 0
            return {"indexedQueries": 0}
    
    def _get_exact_suggestions(self, normalized_input: str, max_suggestions: int) -> List[Dict[str, Any]]:
        """Get suggestions using exact prefix matching."""
        search_params = {
            "yql": f"SELECT * FROM {self.typeahead_schema_name} WHERE query_suffixes contains '{normalized_input}'",
            "hits": max_suggestions,
            "ranking": "default"
        }
        
        response = self.vespa_client.query(schema=self.typeahead_schema_name, **search_params)
        
        if not response or response.status_code != 200:
            return []
        
        hits = response.json().get("root", {}).get("children", [])
        suggestions = []
        
        for hit in hits:
            fields = hit.get("fields", {})
            original_query = fields.get("original_query")
            relevance = hit.get("relevance", 0.0)
            
            if original_query:
                suggestions.append({
                    "query": original_query,
                    "relevance": relevance
                })
        
        return suggestions
    
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
        
        response = self.vespa_client.query(schema=self.typeahead_schema_name, **search_params)
        
        if not response or response.status_code != 200:
            return exact_suggestions
        
        hits = response.json().get("root", {}).get("children", [])
        fuzzy_suggestions = []
        exact_queries = {s["query"] for s in exact_suggestions}
        
        for hit in hits:
            fields = hit.get("fields", {})
            original_query = fields.get("original_query")
            
            if not original_query or original_query in exact_queries:
                continue
            
            # Check if query matches fuzzy criteria
            normalized_query = normalize_text(original_query)
            edit_distance = calculate_edit_distance(normalized_input, normalized_query[:len(normalized_input)])
            
            if edit_distance <= max_edit_distance:
                relevance = hit.get("relevance", 0.0) * (1.0 - edit_distance / (max_edit_distance + 1))
                fuzzy_suggestions.append({
                    "query": original_query,
                    "relevance": relevance
                })
        
        # Combine and sort suggestions
        all_suggestions = exact_suggestions + fuzzy_suggestions
        all_suggestions.sort(key=lambda x: x["relevance"], reverse=True)
        
        return all_suggestions[:max_suggestions]