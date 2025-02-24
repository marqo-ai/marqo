"""
This file contains the `VespaSearchMixin` class, responsible for **executing search queries** against the Vespa engine.

Functions Included:
--------------------
1. **Query Execution**
   - `query()` → Executes a search query using Vespa's Query API.

### **Query Execution Details**
The `query()` method supports:
- **YQL (YQL Query Language)** for structured searches.
- **Ranking profiles** to influence query results.
- **Query features** such as filters, pagination, and timeout settings.

This mixin **requires** `http_client` and `query_url` from `VespaClientBase` to send HTTP requests.

### **Usage Example**
```python
vespa = VespaClient(...)
response = vespa.query()
print(response)
```
"""

from typing import Dict, Any, TYPE_CHECKING

import httpx
import orjson

import marqo.logging
from marqo.vespa.exceptions import VespaError
from marqo.vespa.models import QueryResult

if TYPE_CHECKING:
    from ._client_base import VespaClientBase

logger = marqo.logging.get_logger(__name__)




class VespaSearchMixin:
    def query(
            self: "VespaClientBase", yql: str, hits: int = 10, ranking: str = None, model_restrict: str = None,
            query_features: Dict[str, Any] = None, timeout: float = None, **kwargs) -> QueryResult:
        """
        Query Vespa.
        Args:
            yql: YQL query
            hits: Number of hits to return
            ranking: Ranking profile to use
            model_restrict: Schema to restrict the query to
            query_features: Query features
            **kwargs: Additional query parameters
        Returns:
            Query result as a VespaQueryResult object
        """
        query_features_list = {f'input.query({key})': value for key, value in
            query_features.items()} if query_features else {}

        query = {
            'yql': yql,
            'hits': hits,
            'ranking': ranking,
            'model.restrict': model_restrict, **query_features_list, **kwargs
        }

        # Use default timeout if not already set.
        if timeout:
            query['timeout'] = f"{timeout}ms"
        else:
            query['timeout'] = f"{self.default_search_timeout_ms}ms"

        query = {key: value for key, value in query.items() if value is not None}

        logger.debug(f'Query: {query}')

        try:
            resp = self.http_client.post(f'{self.query_url}/search/', json=query)
        except httpx.HTTPError as e:
            raise VespaError(e) from e

        self._query_raise_for_status(resp)

        return QueryResult(**orjson.loads(resp.text))
