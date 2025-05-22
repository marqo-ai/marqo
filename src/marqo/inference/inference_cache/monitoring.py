from abc import ABC, abstractmethod
from typing import Callable, Iterable

from opentelemetry.metrics import CallbackOptions, Observation

from marqo.otel import metrics


class CacheStatsCollector(ABC):
    """Abstract interface for collecting cache metrics."""

    @abstractmethod
    def record_get(self, hit: bool, duration: float) -> None:
        ...

    @abstractmethod
    def record_insert(self, size_bytes, duration: float) -> None:
        """Called when an item is inserted; `size_bytes` is optional."""
        ...


class OTELCacheStatsCollector(CacheStatsCollector):
    def __init__(self, curr_size_fn: Callable[[], int], max_size_fn: Callable[[], int]):
        meter = metrics.get_meter('inference_cache_stats')

        def get_current_size(options: CallbackOptions) -> Iterable[Observation]:
            yield Observation(curr_size_fn())

        def get_max_size(options: CallbackOptions) -> Iterable[Observation]:
            yield Observation(max_size_fn())

        meter.create_observable_gauge("cache_size_curr", callbacks=[get_current_size],
                                      unit="1", description="Current cache size")
        meter.create_observable_gauge("cache_size_max", callbacks=[get_max_size],
                                      unit="1", description="Current cache size")

        self.hit_counter = meter.create_counter("cache_hits_total", unit="1", description="Total cache hits")
        self.miss_counter = meter.create_counter("cache_miss_total", unit="1", description="Total cache misses")
        self.get_histogram = meter.create_histogram("cache_get_latency", unit="us",
                                                    description="Get latency in microseconds")
        self.insert_histogram = meter.create_histogram("cache_insert_latency", unit="us",
                                                       description="Insertion latency in microseconds")

    def record_get(self, hit: bool, duration: float) -> None:
        if hit:
            self.hit_counter.add(1)
        else:
            self.miss_counter.add(1)

        self.get_histogram.record(duration * 1_000_000)  # in microseconds

    def record_insert(self, size_bytes, duration: float) -> None:
        self.insert_histogram.record(duration * 1_000_000)  # in microseconds
