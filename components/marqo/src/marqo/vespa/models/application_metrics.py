from enum import Enum
from typing import List, Dict, Optional, Union, Any

from pydantic.v1 import BaseModel


class Status(BaseModel):
    code: str
    description: str


class MetricSet(BaseModel):
    dimensions: Dict[str, str]
    values: Dict[str, Any]


class Service(BaseModel):
    name: str
    timestamp: int
    status: Status
    metrics: List[MetricSet]


class Node(BaseModel):
    hostname: str
    role: str
    services: List[Service]


class Aggregation(Enum):
    Max = 'max'
    Min = 'min'
    Average = 'average'
    Sum = 'sum'
    Count = 'count'
    Last = 'last'


# noinspection PyPep8Naming
class ApplicationMetrics(BaseModel):
    nodes: List[Node]

    _SERVICE_CLUSTERCONTROLLER = 'vespa.container-clustercontroller'

    @property
    def clusterController_resourceUsage_maxMemoryUtilization_max(self) -> Optional[Union[int, float]]:
        return self._aggregate_metric(
            metric_name='cluster-controller.resource_usage.max_memory_utilization.max',
            aggregation=Aggregation.Max,
            service_name=self._SERVICE_CLUSTERCONTROLLER
        )

    @property
    def clusterController_resourceUsage_maxDiskUtilization_max(self) -> Optional[Union[int, float]]:
        return self._aggregate_metric(
            metric_name='cluster-controller.resource_usage.max_disk_utilization.max',
            aggregation=Aggregation.Max,
            service_name=self._SERVICE_CLUSTERCONTROLLER
        )

    @property
    def clusterController_resourceUsage_nodesAboveLimit_max(self) -> Optional[Union[int, float]]:
        return self._aggregate_metric(
            metric_name='cluster-controller.resource_usage.nodes_above_limit.max',
            aggregation=Aggregation.Max,
            service_name=self._SERVICE_CLUSTERCONTROLLER
        )

    def documentDb_diskUsage_bytes(self, document_type: str) -> Optional[Union[int, float]]:
        """
        Get the on-disk size, in bytes, of a given document type (Marqo index schema), summed across all
        content nodes.

        Args:
            document_type: The Vespa document type / schema name of the Marqo index

        Returns:
            Total disk usage in bytes for the document type, or None if the metric is not reported
        """
        return self._aggregate_metric(
            metric_name='content.proton.documentdb.documentstore.disk_usage.last',
            aggregation=Aggregation.Sum,
            dimension_filter={'documenttype': document_type}
        )

    def _aggregate_metric(
            self,
            metric_name: str,
            aggregation,
            service_name: Optional[str] = None,
            dimension_filter: Optional[Dict[str, str]] = None
    ) -> Optional[Union[int, float]]:
        """
        Aggregate a metric across all nodes and services.

        Args:
            metric_name: Name of metric to aggregate
            aggregation: Aggregation to use
            service_name: Optional name of service to aggregate metric for. Provioding this will speed up the
                aggregation process
            dimension_filter: Optional dict of dimension name/value pairs that must all match a metric set's
                dimensions for it to be included in the aggregation

        Returns:
            Aggregated metric value. If no values are found and aggregation type is not Aggregation.Count,
            None is returned
        """
        values = []
        for node in self.nodes:
            for service in node.services:
                if service_name is not None and service.name != service_name:
                    continue

                for metric_set in service.metrics:
                    if dimension_filter is not None and not all(
                            metric_set.dimensions.get(key) == value for key, value in dimension_filter.items()
                    ):
                        continue

                    if metric_name in metric_set.values:
                        values.append(metric_set.values[metric_name])

        if aggregation == Aggregation.Count:
            return len(values)

        if len(values) == 0:
            return None

        if aggregation == Aggregation.Max:
            return max(values)
        elif aggregation == Aggregation.Min:
            return min(values)
        elif aggregation == Aggregation.Average:
            return sum(values) / len(values)
        elif aggregation == Aggregation.Sum:
            return sum(values)
        elif aggregation == Aggregation.Last:
            return values[-1]
        else:
            raise ValueError(f"Unknown aggregation {aggregation}")
