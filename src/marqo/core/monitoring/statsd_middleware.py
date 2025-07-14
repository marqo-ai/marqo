import re
import time
from typing import Dict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from marqo.core.monitoring.statsd_client import StatsDClient

_SEARCH_RE = re.compile(r"/indexes/[^/]+/search$")
_DOCS_RE = re.compile(r"/indexes/[^/]+/documents$")
_DOCUMENT_ID_RE = re.compile(r"(/documents/)[^/]+")


class StatsDMiddleware(BaseHTTPMiddleware):
    """
    Emits the former Reverse-Proxy (RP) CloudWatch metrics via StatsD.

    Metrics implemented (parity with RP):
      • requests.completed                counter   • status_code
                                              also  • path, method, status_code
      • marqo_processing_time             timing    (no tags)
      • search_processing_time            timing    (no tags)
      • index_processing_time             timing    (no tags)
      • x-count-success / -failure / -error  counter • method
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

        status = response.status_code
        status_tag = f"{status // 100}XX"      # 2XX / 3XX / 4XX / 5XX

        # --- requests.completed (status-only) ------------------------
        self.statsd.increment("requests.completed", tags={"status_code": status_tag})

        # --- requests.completed (path/method/status variant) ---------
        sanitized_path = self._sanitize_path(request.url.path)
        self.statsd.increment(
            "requests.completed",
            tags={
                "path": sanitized_path,
                "method": request.method,
                "status_code": status_tag,
            },
        )

        # --- marqo_processing_time -----------------------------
        self.statsd.timing("marqo_processing_time", duration_ms)

        # --- search_processing_time ----------------------------
        if _SEARCH_RE.fullmatch(request.url.path):
            self.statsd.timing("search_processing_time", duration_ms)

        # --- index_processing_time and x-count-* counters -------
        if _DOCS_RE.fullmatch(request.url.path):
            if request.method in {"POST", "PATCH"}:
                self.statsd.timing("index_processing_time", duration_ms)

            if request.method in {"POST", "PATCH", "GET"}:
                lowered: Dict[str, str] = {k.lower(): v for k, v in response.headers.items()}
                for hdr in ("x-count-success", "x-count-failure", "x-count-error"):
                    if hdr in lowered:
                        try:
                            self.statsd.increment(hdr, int(lowered[hdr]), tags={"method": request.method})
                        except ValueError:
                            # Header value wasn’t an int – ignore
                            pass

        return response

    @staticmethod
    def _sanitize_path(path: str) -> str:
        """
        Replace the document-id segment in …/documents/{id} with <document_id>
        so we don’t create a high-cardinality CloudWatch dimension.
        """
        return _DOCUMENT_ID_RE.sub(r"\1<document_id>", path)
