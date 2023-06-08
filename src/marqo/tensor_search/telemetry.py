from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from typing import Any, Callable, Dict, List, Optional, Union
import json
from collections import defaultdict
from contextlib import contextmanager
import time
from contextvars import ContextVar
from marqo.tensor_search.tensor_search_logging import get_logger

logger = get_logger(__name__)
current_request: ContextVar[Request] = ContextVar('current_request')

class TimerError(Exception):
    """An error occured whn operating on a metric timer (e.g. stopping a stopped timer)"""
    pass


class Timer:
    def __init__(self):
        self.start_time = None

    def start(self) -> None:
        """Start a new timer"""
        if self.start_time is not None:
            logger.warn(f"'.start()' called on already running timer.")
        else:
            self.start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self.start_time is None:
            logger.warn(f"'.stop()' called on unstarted timer. '.start()' must be called before '.stop()'.")
            raise TimerError()
        else:
            elapsed_time = time.perf_counter() - self.start_time
            self.start_time = None
            return elapsed_time
        

class RequestMetrics():
    METRIC_STORES: Dict[str, "RequestMetrics"] = {}

    @classmethod
    def for_request(cls, r: Optional[Request] = None) -> "RequestMetrics":
        if r is None:
            r = current_request.get()
            
        return cls.METRIC_STORES[r]

    @classmethod
    def set_in_request(cls, r: Request) -> None:
        if r not in cls.METRIC_STORES:
            current_request.set(r)
            cls.METRIC_STORES[r] = RequestMetrics()

    @classmethod
    def clear_metrics_for(cls, r: Request) -> None:
        cls.METRIC_STORES.pop(r, None)


    def __init__(self):
        self.counter: Dict[str, int] = defaultdict(int)
        self.times: Dict[str, float] = defaultdict(float)
        self.timers: Dict[str, Timer] = defaultdict(Timer)

    def increment_counter(self, k: str):
        self.counter[k]+=1

    @contextmanager
    def time(self, k: str):
        self.start(k)
        try:
            yield
        finally:
            elapsed_time = self.stop(k)
            self.times[k] = elapsed_time

    def start(self, k: str):
        """Start a new timer for the given key"""
        self.timers[k].start()

    def stop(self, k: str) -> float:
        """Stop the timer for the given key, and report the elapsed time"""
        try:
            elapsed_time = self.timers[k].stop()
            self.times[k] += elapsed_time
            return elapsed_time
        except TimerError:
            logger.warn(f"timer {k} stopped incorrectly. Time not recorded.")

    def increment_counter(self, k: str, v: int = 1):
        self.counter[k]+=v

    def json(self):
        return {
            "counter": dict(self.counter),
            "times": dict(self.times)
        }

class TelemetryMiddleware(BaseHTTPMiddleware):
    """
    Responsible for starting a request-level metric object, capturing telemetry and injecting 
    it into the Response payload. Metrics are only returned if the `DEFAULT_TELEMETRY_QUERY_PARAM`
    query parameter is provided and it is  "true". Otherwise the request is not altered.    
    """

    DEFAULT_TELEMETRY_QUERY_PARAM = "telemetry"

    def __init__(self, app, **options):    
        self.telemetry_flag: Optional[str] = options.pop("telemetery_flag", TelemetryMiddleware.DEFAULT_TELEMETRY_QUERY_PARAM)
        super().__init__(app, **options)

    def telemetry_enabled_for_request(self, request: Request) -> bool:
        """Returns True if the given request should have metric telemetry recorded and returned in the response."""
        return request.query_params.get(self.telemetry_flag, "false").lower() == "true"

    async def get_response_json(self, response: Response) -> Union[List, Dict]:
        body = b""
        async for chunk in response.body_iterator:
            body += chunk

        return json.loads(body)

    async def dispatch(self, request: Request, call_next: Callable[[], Any]):
        """Wraps the request chain for a given request.

        Args:
            request: The request being processed
            call_next: A callable to the remaining request call-chain.

        """

        RequestMetrics.set_in_request(request)

        response = await call_next(request)

        # Early exit if opentelemetry is not to be injected into response.
        if not self.telemetry_enabled_for_request(request):
            return response

        data = await self.get_response_json(response)
        
        # Inject telemetry and fix content-length header
        data["telemetry"] = RequestMetrics.for_request(request).json()
        RequestMetrics.clear_metrics_for(request)

        body = json.dumps(data).encode()
        response.headers["content-length"] = str(len(body))
        
        return Response(
            content=body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type
        )