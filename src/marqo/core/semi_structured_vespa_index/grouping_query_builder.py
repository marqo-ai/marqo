"""
Vespa grouping query builder for variant deduplication

This module provides functionality to transform standard YQL queries into 
grouping queries that deduplicate results based on a specified field.
"""

import re
from typing import Optional


class GroupingQueryBuilder:
    """Builder for Vespa grouping queries to handle variant deduplication"""
    
    @staticmethod
    def build_grouping_query(
        base_yql: str, 
        group_field: str, 
        max_per_group: int = 1
    ) -> str:
        """
        Transform a base YQL query to include grouping for variant deduplication.
        
        Args:
            base_yql: The original YQL query (e.g., "select * from sources where ...")
            group_field: Field name to group by (e.g., "product_id")
            max_per_group: Maximum number of results per group
            
        Returns:
            Modified YQL query with grouping clause
            
        Example:
            Input: "select * from sources where title contains 'shoes'"
            Output: "select * from sources where title contains 'shoes' | all(group(product_id) max(1) each(output(summary())))"
        """
        if not base_yql or not group_field:
            raise ValueError("base_yql and group_field are required")
            
        if max_per_group < 1:
            raise ValueError("max_per_group must be at least 1")
        
        # Clean the base YQL
        base_yql = base_yql.strip()

        grouping_clause = f"all(group({group_field}) each(max({max_per_group}) each(output(summary()))))"

        # return f"{base_yql} limit 0 | {grouping_clause}"

        return base_yql
    
    @staticmethod
    def extract_base_query(grouped_yql: str) -> Optional[str]:
        """
        Extract the base query from a grouped YQL query.
        
        Args:
            grouped_yql: YQL query with grouping clause
            
        Returns:
            Base query without grouping, or None if not a grouped query
        """
        if '|' not in grouped_yql:
            return None
            
        # Split on the first pipe and return the base part
        parts = grouped_yql.split('|', 1)
        return parts[0].strip()
    
    @staticmethod
    def is_grouped_query(yql: str) -> bool:
        """
        Check if a YQL query contains grouping.
        
        Args:
            yql: YQL query to check
            
        Returns:
            True if the query contains grouping syntax
        """
        # Look for grouping patterns
        grouping_patterns = [
            r'all\s*\(\s*group\s*\(',
            r'each\s*\(\s*group\s*\(',
        ]
        
        for pattern in grouping_patterns:
            if re.search(pattern, yql, re.IGNORECASE):
                return True
                
        return False
    
    @staticmethod
    def validate_grouping_query(yql: str) -> bool:
        """
        Validate that a grouping query is syntactically correct.
        
        Args:
            yql: YQL query to validate
            
        Returns:
            True if the query appears to be valid
        """
        if not yql:
            return False
            
        # Basic validation - check for balanced parentheses
        open_parens = yql.count('(')
        close_parens = yql.count(')')
        
        if open_parens != close_parens:
            return False
            
        # Check for required grouping components if it's a grouped query
        if GroupingQueryBuilder.is_grouped_query(yql):
            # Should have group(...) and output(...)
            has_group = re.search(r'group\s*\([^)]+\)', yql, re.IGNORECASE)
            has_output = re.search(r'output\s*\([^)]*summary[^)]*\)', yql, re.IGNORECASE)
            
            return bool(has_group and has_output)
            
        return True