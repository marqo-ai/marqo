from typing import Callable

from fastapi import FastAPI
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    MetricExportResult,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import SERVICE_NAME, Resource

from inference_orchestrator.core.logging import get_logger
from inference_orchestrator.core.settings import get_settings

settings = get_settings()


class LoggingMetricExporter(ConsoleMetricExporter):
    """A MetricExporter that logs via the Python logging system instead of printing."""

    def __init__(self, logger_name: str = "metrics"):
        super().__init__()
        self.logger = get_logger(logger_name)

    def export(
        self, metrics_data, timeout_millis: float = 10_000, **kwargs
    ) -> MetricExportResult:
        self.logger.info(metrics_data.to_json(indent=None))
        return MetricExportResult.SUCCESS


def bootstrap_otel(app: FastAPI, service_name: str) -> Callable[[], None]:
    export_interval_seconds = settings.marqo_metrics_export_interval
    if export_interval_seconds == 0:
        # disable metrics export, and return no_op as shutdown hook
        return lambda: None

    exporter = LoggingMetricExporter()
    reader = PeriodicExportingMetricReader(
        exporter, export_interval_millis=export_interval_seconds * 1000
    )

    resource = Resource({SERVICE_NAME: service_name})
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(meter_provider)

    # TODO add default instrumentation of app

    def shutdown_hook() -> None:
        meter_provider.shutdown()

    return shutdown_hook
