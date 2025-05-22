from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    MetricExportResult,
    PeriodicExportingMetricReader,
)

from marqo import logging


class LoggingMetricExporter(ConsoleMetricExporter):
    """A MetricExporter that logs via the Python logging system instead of printing."""
    def __init__(self, logger_name: str = "metrics"):
        super().__init__()
        self.logger = logging.get_logger(logger_name)

    def export(self, metrics_data, timeout_millis: float = 10_000, **kwargs) -> MetricExportResult:
        self.logger.info(metrics_data.to_json(indent=None))
        return MetricExportResult.SUCCESS


exporter = LoggingMetricExporter()
reader = PeriodicExportingMetricReader(exporter, export_interval_millis=60_000)
metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
