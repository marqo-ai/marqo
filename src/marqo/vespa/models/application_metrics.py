from enum import Enum
from typing import List, Dict, Optional, Union, Any

from pydantic import BaseModel


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


class ApplicationMetrics(BaseModel):
    nodes: List[Node]

    def aggregate_metric(
            self,
            metric_name: str,
            aggregation,
            service_name: Optional[str] = None
    ) -> Optional[Union[int, float]]:
        """
        Aggregate a metric across all nodes and services.

        Args:
            metric_name: Name of metric to aggregate
            aggregation: Aggregation to use
            service_name: Optional name of service to aggregate metric for. Provioding this will speed up the
                aggregation process

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
