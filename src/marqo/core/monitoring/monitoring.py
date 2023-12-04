from typing import Optional

import marqo.logging
from marqo.core.exceptions import IndexNotFoundError
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models import MarqoIndex
from marqo.core.models.marqo_index_health import MarqoHealthStatus, HealthStatus, VespaHealthStatus
from marqo.core.models.marqo_index_stats import MarqoIndexStats
from marqo.core.vespa_index import for_marqo_index as vespa_index_factory
from marqo.exceptions import InternalError
from marqo.vespa.exceptions import VespaError
from marqo.vespa.vespa_client import VespaClient

logger = marqo.logging.get_logger(__name__)


class Monitoring:

    def __init__(self, vespa_client: VespaClient, index_management: IndexManagement):
        self.vespa_client = vespa_client
        self.index_management = index_management

    def get_index_stats(self, marqo_index: MarqoIndex) -> MarqoIndexStats:
        """
        Get statistics for a Marqo index.
        Args:
            marqo_index: Marqo index to get statistics for

        Returns:
            Marqo index statistics
        """
        vespa_index = vespa_index_factory(marqo_index)

        doc_count_query_result = self.vespa_client.query(
            yql=f'select * from {marqo_index.name} where true limit 0',
            model_restrict=marqo_index.name
        )
        vector_count_query_result = self.vespa_client.query(
            **vespa_index.get_vector_count_query()
        )

        try:
            if vector_count_query_result.root.children[0].children is None:  # empty index
                number_of_vectors = 0
            else:
                number_of_vectors = list(
                    vector_count_query_result.root.children[0].children[0].children[0].fields.values()
                )[0]
        except (TypeError, AttributeError, IndexError) as e:
            raise InternalError(f"Failed to get the number of vectors for index {marqo_index.name}: {e}") from e

        return MarqoIndexStats(
            number_of_documents=doc_count_query_result.total_count,
            number_of_vectors=number_of_vectors
        )

    def get_index_stats_by_name(self, index_name: str) -> MarqoIndexStats:
        """
        Get statistics for a Marqo index.

        Args:
            index_name: Name of Marqo index to get statistics for

        Returns:
            Marqo index statistics
        """
        marqo_index = self.index_management.get_index(index_name)
        return self.get_index_stats(marqo_index)

    def get_health(self, index_name: Optional[str] = None, hostname_filter: Optional[str] = None) -> MarqoHealthStatus:
        """
        Get health status for a Marqo index.

        Args:
            index_name: Optional name of Marqo index to get health status for
            hostname_filter: Optional hostname filter. If provided, only Vespa nodes with this value in their hostname
            will be considered in the health check

        Returns:
            Marqo index health status
        """
        # TODO - Check index specific metrics such as memory and disk usage
        marqo_status = self._get_marqo_health()
        try:
            vespa_status = self._get_vespa_health(hostname_filter=hostname_filter)
        except VespaError as e:
            logger.warning(f"Failed to get Vespa health: {e}")
            vespa_status = HealthStatus.Red

        aggregated_status = max(marqo_status, vespa_status)

        return MarqoHealthStatus(
            status=aggregated_status,
            backend=VespaHealthStatus(status=vespa_status)
        )

    def _get_marqo_health(self) -> HealthStatus:
        return HealthStatus.Green

    def _get_vespa_health(self, hostname_filter: Optional[str]) -> HealthStatus:
        metrics = self.vespa_client.get_metrics()

        status = HealthStatus.Green
        for node in metrics.nodes:
            if hostname_filter is not None and hostname_filter not in node.hostname:
                continue

            for service in node.services:
                if service.status.code != 'up':
                    status = HealthStatus.Red
                    break

        return status
