import re
import time
from typing import Dict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from pathlib import PurePosixPath

from marqo.core.monitoring.statsd_client import StatsDClient

_DOCS_RE = re.compile(r"/indexes/[^/]+/documents$")

FIXED_DOC_SUBPATHS = {"delete-batch", "get-batch"}


class StatsDMiddleware(BaseHTTPMiddleware):
    """
    Emits high-cardinality generic metrics.

    • request.duration_ms   |ms  path,method,status_code
    • batch.success         |c   path,method,status_code
    • batch.failure         |c   path,method,status_code
    • batch.error           |c   path,method,status_code
    """

    def __init__(self, app, statsd_client: StatsDClient):
        super().__init__(app)
        self.statsd = statsd_client

    async def dispatch(self, request: Request, call_next):
        # -- short-circuit: don’t record metrics for the health-check -----------
        if request.url.path == "/":
            return await call_next(request)
        t_start = time.perf_counter()
        response: Response = await call_next(request)
        duration_ms = int((time.perf_counter() - t_start) * 1000)

        path_tag = self._sanitize_path(request.url.path)
        tags = {
            "path": path_tag,
            "method": request.method,
            "status_code": str(int(response.status_code)),
        }

        # latency
        self.statsd.timing("request.duration_ms", duration_ms, tags=tags)

        # batch outcome counters
        if _DOCS_RE.fullmatch(request.url.path) and request.method in {"POST", "PATCH", "GET"}:
            lowered: Dict[str, str] = {k.lower(): v for k, v in response.headers.items()}
            for hdr, metric in (
                    ("x-count-success", "batch.success"),
                    ("x-count-failure", "batch.failure"),
                    ("x-count-error", "batch.error"),
            ):
                if hdr in lowered:
                    try:
                        self.statsd.increment(metric, int(lowered[hdr]), tags=tags)
                    except ValueError:
                        # Header value wasn’t an int – ignore
                        pass

        return response

    @staticmethod
    def _sanitize_path(path: str) -> str:
        """
        Replace the document-id segment in …/documents/{id} with <document_id>,
        but skip known fixed subpaths like …/documents/delete-batch.
        """
        p = PurePosixPath(path)
        if p.parent.name == "documents" and p.name not in FIXED_DOC_SUBPATHS:
            return str(p.parent / "<document_id>")
        return path