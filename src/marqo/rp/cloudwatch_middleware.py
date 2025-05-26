import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from marqo.rp import cloudwatch_metrics

TRACKED_ENDPOINTS = [
    ("/indexes/", "/search", "POST"),
    ("/indexes/", "/documents", "POST"),
]

class CloudWatchMetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        method = request.method

        should_track = any(
            path.startswith(prefix) and path.endswith(suffix) and method == expected_method
            for prefix, suffix, expected_method in TRACKED_ENDPOINTS
        )

        if not should_track:
            return await call_next(request)

        t0 = time.time()
        response: Response = await call_next(request)
        duration = (time.time() - t0) * 1000

        cloudwatch_metrics.add_metric(
            name="RequestLatency",
            value=duration,
            extra_dims={
                "Path": path,
                "Method": method,
                "StatusCode": str(response.status_code),
            }
        )

        return response
