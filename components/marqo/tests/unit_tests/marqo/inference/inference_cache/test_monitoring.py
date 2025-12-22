import unittest
from typing import Any, Dict

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics._internal.export import InMemoryMetricReader
from opentelemetry.sdk.metrics._internal.point import Metric
from opentelemetry.test.globals_test import reset_metrics_globals

from marqo.core.inference.inference_cache.monitoring import OTELCacheStatsCollector


class TestOTELCacheStatsCollector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reset_metrics_globals()
        cls.reader = InMemoryMetricReader()
        cls.provider = MeterProvider(metric_readers=[cls.reader])
        metrics.set_meter_provider(cls.provider)

    @classmethod
    def tearDownClass(cls):
        cls.provider.shutdown()

    def setUp(self):
        self.stats_collector = OTELCacheStatsCollector(
            curr_size_fn=lambda: 42,
            max_size_fn=lambda: 99,
        )

    def test_otel_stats(self):
        self.stats_collector.record_get(True, 0.0009)
        self.stats_collector.record_get(True, 0.0008)
        self.stats_collector.record_get(False, 0.0004)
        self.stats_collector.record_set(100, 0.001)
        self.stats_collector.record_set(200, 0.0005)

        self.reader.force_flush()

        metrics_data = self.reader.get_metrics_data()
        self.assertEqual(metrics_data.resource_metrics[0].scope_metrics[0].scope.name, 'inference_cache_stats')

        metrics = metrics_data.resource_metrics[0].scope_metrics[0].metrics
        self.assertEqual(7, len(metrics))
        self._assert_single_value(metrics[0], 'cache_hit_total', 2)
        self._assert_single_value(metrics[2], 'cache_miss_total', 1)
        self._assert_single_value(metrics[3], 'insert_item_size', 300)
        self._assert_single_value(metrics[5], 'cache_size_curr', 42)
        self._assert_single_value(metrics[6], 'cache_size_max', 99)

        # duration we record is in seconds, and the metric we store in the latency histogram is in microseconds(us)
        # default buckets [0.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0, 250.0, 500.0, 750.0, 1000.0, 2500.0, 5000.0, 7500.0, 10000.0]
        self._assert_histogram(metrics[1], 'cache_get_latency', bucket_counts={500.0: 1, 1000.0: 2})
        self._assert_histogram(metrics[4], 'cache_set_latency', bucket_counts={500.0: 1, 1000.0: 1})

    def _assert_single_value(self, metric: Metric, name: str, value: Any):
        self.assertEqual(metric.name, name)
        self.assertEqual(metric.data.data_points[0].value, value)

    def _assert_histogram(self, metric: Metric, name: str, bucket_counts: Dict[float, int]):
        self.assertEqual(metric.name, name)
        explicit_bounds = metric.data.data_points[0].explicit_bounds
        for index, bucket in enumerate(explicit_bounds):
            actual_value = metric.data.data_points[0].bucket_counts[index]
            if bucket in bucket_counts:
                self.assertEqual(
                    actual_value, bucket_counts[bucket],
                    f'histogram {name} bucket {bucket} should be {bucket_counts[bucket]}, but was {actual_value}')
            else:
                self.assertEqual(actual_value, 0,
                                 f'histogram {name} bucket {bucket} should be 0, but was {actual_value}')

        for bucket in bucket_counts.keys():
            if bucket not in explicit_bounds:
                self.fail(f'histogram {name} bucket {bucket} is not in metrics: {explicit_bounds}')
