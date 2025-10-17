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
    def record_set(self, size_bytes: int, duration: float) -> None:
        """Called when an item is set"""
        ...


class OTELCacheStatsCollector(CacheStatsCollector):
    def __init__(self, curr_size_fn: Callable[[], int], max_size_fn: Callable[[], int]):
        meter = metrics.get_meter('inference_cache_stats')

        def get_current_size(options: CallbackOptions) -> Iterable[Observation]:
            curr_size = curr_size_fn()
            yield Observation(curr_size)

        def get_max_size(options: CallbackOptions) -> Iterable[Observation]:
            max_size = max_size_fn()
            yield Observation(max_size)

        meter.create_observable_gauge("cache_size_curr", callbacks=[get_current_size],
                                      unit="1", description="Current cache size")
        meter.create_observable_gauge("cache_size_max", callbacks=[get_max_size],
                                      unit="1", description="Current cache size")

        self.hit_counter = meter.create_counter("cache_hit_total", unit="1", description="Total cache hits")
        self.miss_counter = meter.create_counter("cache_miss_total", unit="1", description="Total cache misses")
        self.item_size_counter = meter.create_counter("insert_item_size", unit="byte", description="Item size")

        # duration we record is in seconds, and the metric we store in the latency histogram is in microseconds(us)
        # default buckets are [0.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0, 250.0, 500.0, 750.0, 1000.0, 2500.0, 5000.0, 7500.0, 10000.0]
        self.get_histogram = meter.create_histogram("cache_get_latency", unit="us",
                                                    description="Get latency in microseconds")
        self.insert_histogram = meter.create_histogram("cache_set_latency", unit="us",
                                                       description="Set latency in microseconds")

    def record_get(self, hit: bool, duration: float) -> None:
        if hit:
            self.hit_counter.add(1)
        else:
            self.miss_counter.add(1)

        self.get_histogram.record(duration * 1_000_000)  # in microseconds

    def record_set(self, size_bytes: int, duration: float) -> None:
        self.item_size_counter.add(size_bytes)
        self.insert_histogram.record(duration * 1_000_000)  # in microseconds
