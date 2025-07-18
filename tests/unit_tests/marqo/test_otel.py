import unittest
from typing import Callable
from unittest.mock import MagicMock, patch

import opentelemetry.metrics as metrics
from fastapi import FastAPI
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics._internal.export import ConsoleMetricExporter, MetricExportResult
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME
from opentelemetry.test.globals_test import reset_metrics_globals

from marqo.otel import LoggingMetricExporter, bootstrap_otel
from marqo.tensor_search.enums import EnvVars


class TestLoggingMetricExporter(unittest.TestCase):
    def test_construction_with_default_logger_name(self):
        exporter = LoggingMetricExporter()
        # Should inherit from ConsoleMetricExporter
        self.assertTrue(isinstance(exporter, ConsoleMetricExporter))
        # Default logger name
        self.assertEqual(exporter.logger.name, 'metrics')

    def test_export_logs_info_and_returns_success(self):
        exporter = LoggingMetricExporter()
        # Create dummy metrics_data with to_json
        dummy = MagicMock()
        dummy.to_json.return_value = '{"key": "value"}'

        # Capture logging output
        with self.assertLogs(exporter.logger, level='INFO') as cm:
            result = exporter.export(dummy, timeout_millis=5000, extra='param')

        # Verify log content and return value
        self.assertIn('{"key": "value"}', cm.output[0])
        self.assertEqual(result, MetricExportResult.SUCCESS)


class TestBootstrapOtel(unittest.TestCase):
    def setUp(self):
        # Reset global meter provider to a dummy before each test
        reset_metrics_globals()
        self.app = FastAPI()

    def test_meter_provider_setup(self):
        bootstrap_otel(self.app, 'test_service')
        provider = metrics.get_meter_provider()

        # Check provider type and resource
        self.assertIsInstance(provider, MeterProvider)
        self.assertEqual(provider._sdk_config.resource.attributes[SERVICE_NAME], 'test_service')

        # Check reader setup
        readers = provider._sdk_config.metric_readers
        self.assertEqual(len(readers), 1)
        reader = readers[0]
        self.assertIsInstance(reader, PeriodicExportingMetricReader)
        # Export interval should match updated value
        self.assertEqual(reader._export_interval_millis, 30000)
        # Exporter instance
        self.assertIsInstance(reader._exporter, LoggingMetricExporter)

    def test_shutdown_hook(self):
        shutdown_hook = bootstrap_otel(self.app, 'test_service')
        self.assertIsInstance(shutdown_hook, Callable)

        shutdown_hook()

        provider = metrics.get_meter_provider()
        self.assertTrue(provider._shutdown)

    def test_export_interval_millis_is_configurable(self):
        with patch.dict('os.environ', {EnvVars.MARQO_METRICS_EXPORT_INTERVAL: "10"}):
            bootstrap_otel(self.app, 'test_service')
            provider = metrics.get_meter_provider()
            reader = provider._sdk_config.metric_readers[0]
            self.assertEqual(reader._export_interval_millis, 10000)

    def test_otel_export_is_disabled_if_env_var_set_to_zero(self):
        with patch.dict('os.environ', {EnvVars.MARQO_METRICS_EXPORT_INTERVAL: "0"}):
            shutdown_hook = bootstrap_otel(self.app, 'test_service')

            # assert that meter provider is not configured when disabled
            provider = metrics.get_meter_provider()
            self.assertIsNone(provider._real_meter_provider)
            self.assertFalse(hasattr(provider, '_sdk_config'))

            self.assertIsInstance(shutdown_hook, Callable)
            # assert that the shutdown_hook is no_op when disabled
            self.assertEqual(shutdown_hook.__code__.co_code, (lambda: None).__code__.co_code)