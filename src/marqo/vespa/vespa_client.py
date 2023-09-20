from typing import Dict, Any

import requests
from requests.adapters import HTTPAdapter

from marqo.vespa.models.query_result import QueryResult


class VespaClient:
    def __init__(self, config_url: str, document_url: str, query_url: str, pool_size: int = 10):
        """
        Create a VespaClient object.
        Args:
            config_url: Vespa Deploy API base URL
            document_url: Vespa Document API base URL
            query_url: Vespa Query API base URL
            pool_size: Number of connections to keep in the connection pool
        """
        self.config_url = config_url
        self.document_url = document_url
        self.query_url = query_url
        self.session = self._create_session(pool_size)

    def close(self):
        """
        Close the VespaClient object.
        """
        self.session.close()

    def _create_session(self, pool_size: int):
        session = requests.Session()
        adapter = HTTPAdapter(pool_maxsize=pool_size)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def deploy_application(self, application_zip: str):
        """
        Deploy a Vespa application.
        Args:
            application_zip: Path to the Vespa application zip file
        """
        pass

    def query(self, yql: str, hits: int = 10, ranking: str = None, model_restrict: str = None,
              query_features: Dict[str, Any] = None, **kwargs) -> VespaQueryResult:
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
        query_features_list = {
            f'input.query({key})': value for key, value in query_features.items()
        } if query_features else {}
        query = {
            'yql': yql,
            'hits': hits,
            'ranking': ranking,
            'model.restrict': model_restrict,
            **query_features_list,
            **kwargs
        }
        resp = self.session.post(f'{self.query_url}/search/', data=query)

        resp.raise_for_status()

        return QueryResult(**resp.json())
