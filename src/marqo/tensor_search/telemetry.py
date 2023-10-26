from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from typing import Any, Callable, Dict, List, Optional, Union
import json
from collections import defaultdict
from contextlib import contextmanager
import time
from contextvars import ContextVar
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from marqo.tensor_search.tensor_search_logging import get_logger

logger = get_logger(__name__)


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
        """Stop the timer, and report the elapsed time
        
        Return time is in Ms.
        """
        if self.start_time is None:
            raise TimerError(
                f"'.stop()' called on unstarted timer. '.start()' must be called before '.stop()'."
            )
        else:
            elapsed_time = time.perf_counter() - self.start_time
            self.start_time = None
            return 1000 * elapsed_time


class RequestMetrics:
    @classmethod
    def reduce_from_list(cls, metrics: List["RequestMetrics"]) -> "RequestMetrics":
        assert len(metrics) > 0, "Cannot create RequestMetrics from []"
        m = metrics.pop(0)
        for mm in metrics:
            for k, count in mm.counter.items():
                m.increment_counter(k, count)

            for k, timer in mm.timers.items():
                m.timers[k] = timer

            for k, time in mm.times.items():
                m.add_time(k, time)
        return m

    def __init__(self):
        self.counter: Dict[str, int] = defaultdict(int)

        # Note: We default to float, but on multiple times for the same key, it gets converted to a List.
        self.times: Dict[str, Union[float, List[float]]] = defaultdict(float)
        self.timers: Dict[str, Timer] = defaultdict(Timer)

    def increment_counter(self, k: str):
        self.counter[k]+=1

    @contextmanager
    def time(self, k: str, callback: Optional[Callable[[float], None]] = None):
        self.start(k)
        try:
            yield
        finally:
            elapsed_time = self._stop(k)
            if callback is not None:
                callback(elapsed_time)
            self.add_time(k, elapsed_time)

    def start(self, k: str):
        """Start a new timer for the given key"""
        self.timers[k].start()

    def _stop(self, k: str) -> float:
        return self.timers[k].stop()

    def add_time(self, k: str, v: float):
        if self.times.get(k, None) is None:
            self.times[k] = v
        elif isinstance(self.times.get(k), list):
            self.times[k] = self.times[k] + [v]
        else:
            self.times[k] = [self.times[k], v]

    def stop(self, k: str) -> float:
        """Stop the timer for the given key, and report the elapsed time"""
        try:
            elapsed_time = self._stop(k)
            self.add_time(k, elapsed_time)
            return elapsed_time
        except TimerError:
            logger.warn(f"timer {k} stopped incorrectly. Time not recorded.")

    def increment_counter(self, k: str, v: int = 1):
        self.counter[k]+=v

    def json(self):
        return {
            "counter": dict(self.counter),
            "timesMs": dict(self.times)
        }


class RequestMetricsStore():
    current_request: ContextVar[Request] = ContextVar('current_request')
    
    METRIC_STORES: Dict[Request, RequestMetrics] = {}

    @classmethod
    def _set_request(cls, r: Request):
        cls.current_request.set(r)

    @classmethod
    def _get_request(cls) -> Request:
        return cls.current_request.get()

    @classmethod
    def for_request(cls, r: Optional[Request] = None) -> RequestMetrics:
        if r is None:
            r = cls._get_request()
            
        return cls.METRIC_STORES[r]

    @classmethod
    def set_in_request(cls, r: Optional[Request] = None, metrics: Optional[RequestMetrics] = None) -> None:
        """
        NOTE: this function should only be used in TelemetryMiddleware, and threading edge cases.
        """
        r = r if r is not None else cls._get_request()
        cls._set_request(r)
        cls.METRIC_STORES[r] = metrics if metrics is not None else RequestMetrics()

    @classmethod
    def clear_metrics_for(cls, r: Request) -> None:
        cls.METRIC_STORES.pop(r, None)


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
        RequestMetricsStore.set_in_request(request)

        response = await call_next(request)

        # Early exit if opentelemetry is not to be injected into response.
        if not self.telemetry_enabled_for_request(request):
            return response

        data = await self.get_response_json(response)

        # Inject telemetry and fix content-length header
        if isinstance(data, dict):
            telemetry = RequestMetricsStore.for_request(request).json()
            if len(telemetry["timesMs"]) == 0:
                telemetry.pop("timesMs")
            if len(telemetry["counter"]) == 0:
                telemetry.pop("counter")
            data["telemetry"] = telemetry
        else:
            get_logger(__name__).warning(
                f"{self.telemetry_flag} set but response payload is not Dict. telemetry not returned"
            )
            get_logger(__name__).info(f"Telemetry data={json.dumps(RequestMetricsStore.for_request(request).json(), indent=2)}")

        RequestMetricsStore.clear_metrics_for(request)

        body = json.dumps(data).encode()
        response.headers["content-length"] = str(len(body))
        
        return Response(
            content=body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type
        )