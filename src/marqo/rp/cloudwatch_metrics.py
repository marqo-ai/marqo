import os
import time
import threading
import logging
from datetime import datetime
import boto3

log = logging.getLogger(__name__)

AWS_REGION = os.environ.get("APPLICATION_AWS_REGION", "us-east-1")
CLOUDWATCH_NAMESPACE = os.environ.get("CLOUDWATCH_NAMESPACE", "Marqo")
SYS_ACC_ID = os.environ.get("SYS_ACC_ID", "gje7jbgi")
INDEX_NAME = os.environ.get("INDEX_NAME", "test")

client = boto3.client("cloudwatch", region_name=AWS_REGION)

_metrics = []
_lock = threading.Lock()


def add_metric(name: str, value: float, unit: str = "Milliseconds", extra_dims: dict = {}):
    with _lock:
        _metrics.append({
            "MetricName": name,
            "Timestamp": datetime.utcnow(),
            "Value": value,
            "Unit": unit,
            "Dimensions": [
                              {"Name": "SystemAccountId", "Value": SYS_ACC_ID},
                              {"Name": "IndexName", "Value": INDEX_NAME},
                          ] + [{"Name": k, "Value": v} for k, v in extra_dims.items()]
        })


def _flush_metrics():
    global _metrics
    with _lock:
        if not _metrics:
            return
        batch = _metrics[:20]
        _metrics = _metrics[20:]
    try:
        client.put_metric_data(Namespace=CLOUDWATCH_NAMESPACE, MetricData=batch)
    except Exception as e:
        log.error(f"Failed to send metrics: {e}")


def _metrics_loop():
    while True:
        try:
            _flush_metrics()
        except Exception:
            pass
        time.sleep(10)


def start_metrics_loop():
    t = threading.Thread(target=_metrics_loop, daemon=True)
    t.start()
