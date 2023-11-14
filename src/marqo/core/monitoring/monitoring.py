from typing import Optional

import marqo.logging
from marqo.core.exceptions import IndexNotFoundError
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models.marqo_index_health import MarqoHealthStatus, HealthStatus, VespaHealthStatus
from marqo.core.models.marqo_index_stats import MarqoIndexStats
from marqo.vespa.exceptions import VespaError
from marqo.vespa.vespa_client import VespaClient

logger = marqo.logging.get_logger(__name__)


class Monitoring:

    def __init__(self, vespa_client: VespaClient, index_management: IndexManagement):
        self.vespa_client = vespa_client
        self.index_management = index_management

    def get_index_stats(self, index_name: str) -> MarqoIndexStats:
        """
        Get statistics for a Marqo index.
        Args:
            index_name: Name of Marqo index to get statistics for

        Returns:
            Marqo index statistics
        """
        if not self.index_management.index_exists(index_name):
            raise IndexNotFoundError(f"Index {index_name} not found")

        query_result = self.vespa_client.query(yql=f'select * from {index_name} where True limit 0')

        return MarqoIndexStats(
            number_of_documents=query_result.total_count
        )

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
