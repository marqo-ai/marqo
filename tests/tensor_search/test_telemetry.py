import json
import unittest
from unittest.mock import patch

from fastapi import Request
from starlette.applications import Starlette
from starlette.testclient import TestClient
from starlette.responses import JSONResponse
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.datastructures import QueryParams
from starlette.applications import Starlette
from starlette.responses import Response

from marqo.tensor_search.telemetry import RequestMetricsStore, TelemetryMiddleware, Timer, TimerError


class TestTimer(unittest.TestCase):

    def setUp(self):
        self.timer = Timer()

    @patch('time.perf_counter')
    def test_timer_start_stop(self, mock_time):
        # Simulate the time progression
        mock_time.side_effect = [0.0, 1.0]
        
        # Test initial start of timer
        self.timer.start()
        self.assertEqual(self.timer.start_time, 0.0)
        
        # Test warning if starting again without stopping
        with self.assertLogs(level='WARNING') as cm:
            self.timer.start()
        self.assertIn("'.start()' called on already running timer.", cm.output[0])

        # Test stopping of timer
        elapsed_time = self.timer.stop()
        self.assertEqual(elapsed_time, 1000.0)  # returns ms
        self.assertIsNone(self.timer.start_time)
        
        # Test warning if stopping without starting
        with self.assertRaises(TimerError):
            self.timer.stop()

    @patch('time.perf_counter')
    def test_timer_restart(self, mock_time):
        # Simulate the time progression, seconds
        mock_time.side_effect = [0.0, 1.0, 2.0, 4.0]
        
        # Test timer start, stop, and restart
        self.timer.start()
        elapsed_time = self.timer.stop()
        self.assertEqual(elapsed_time, 1000.0) # returns ms
        
        self.timer.start()
        elapsed_time = self.timer.stop()
        self.assertEqual(elapsed_time, 2000.0) # returns ms


class TestRequestMetricsStore(unittest.TestCase):

    def setUp(self):
        RequestMetricsStore.METRIC_STORES = {} # hard clear
        self.request = Request(scope={"type": "http"})

    def tearDown(self) -> None:
        RequestMetricsStore.clear_metrics_for(self.request)

    def test_set_in_request_and_for_request(self):
        self.assertEqual({}, RequestMetricsStore.METRIC_STORES)

        RequestMetricsStore.set_in_request(self.request)
        self.assertIsNotNone(RequestMetricsStore.for_request(self.request))

    def test_clear_metrics_for(self):
        RequestMetricsStore.set_in_request(self.request)
        self.assertIsNotNone(RequestMetricsStore.for_request(self.request))

        RequestMetricsStore.clear_metrics_for(self.request)
        self.assertEqual({}, RequestMetricsStore.METRIC_STORES)

    @patch.object(Timer, 'start')
    @patch.object(Timer, 'stop')
    def test_time(self, mock_timer_stop, mock_timer_start):
        RequestMetricsStore.set_in_request(r=self.request) # As set by TelemetryMiddleware
        
        mock_timer_stop.return_value = 1.0
        key = 'timer1'
        metric = RequestMetricsStore.for_request(self.request)

        with metric.time(key):
            pass

        self.assertEqual(
            metric.json(),
            {
                "counter": {},
                "timesMs": {key: 1.0}
            }
        )
        mock_timer_start.assert_called_once()
        mock_timer_stop.assert_called_once()

    @patch.object(Timer, 'start')
    def test_start(self, mock_timer_start):
        RequestMetricsStore.set_in_request(r=self.request) # As set by TelemetryMiddleware
        
        metric = RequestMetricsStore.for_request(self.request)
        metric.start('timer1')
        mock_timer_start.assert_called_once()

    @patch.object(Timer, 'stop')
    def test_stop_success(self, mock_timer_stop):
        RequestMetricsStore.set_in_request(r=self.request) # As set by TelemetryMiddleware
        
        mock_timer_stop.return_value = 1.0
        key = 'timer1'
        metric = RequestMetricsStore.for_request(self.request)
        elapsed_time = metric.stop(key)
        self.assertEqual(
            metric.json(),
            {
                "counter": {},
                "timesMs": {key: 1.0}
            }
        )
        self.assertEqual(1.0, elapsed_time)
        mock_timer_stop.assert_called_once()

    def test_increment_counter_with_value(self):
        RequestMetricsStore.set_in_request(r=self.request) # As set by TelemetryMiddleware

        key = 'counter1'
        value = 5.0
        metric = RequestMetricsStore.for_request(self.request)
        metric.increment_counter(key, v=value)
        self.assertEqual(metric.counter, {key: value})

    def test_increment_counter_multiple_times(self):
        RequestMetricsStore.set_in_request(r=self.request) # As set by TelemetryMiddleware

        key = 'counter1'
        metric = RequestMetricsStore.for_request(self.request)
        metric.increment_counter(key)
        metric.increment_counter(key)
        self.assertEqual(metric.counter, {key: 2})

    @patch.object(Timer, 'start')
    @patch.object(Timer, 'stop')
    def test_time_with_exception(self, mock_timer_stop, mock_timer_start):
        RequestMetricsStore.set_in_request(r=self.request) # As set by TelemetryMiddleware

        mock_timer_stop.return_value = 1.0
        key = 'timer1'
        metric = RequestMetricsStore.for_request(self.request)

        with self.assertRaises(Exception):
            with metric.time(key):
                raise Exception("Test exception")

        self.assertEqual(
            metric.json(),
            {
                "counter": {},
                "timesMs": {key: 1.0}
            }
        )
        mock_timer_start.assert_called_once()
        mock_timer_stop.assert_called_once()

    @patch.object(Timer, 'stop')
    def test_stop_without_start(self, mock_timer_stop):
        RequestMetricsStore.set_in_request(r=self.request) # As set by TelemetryMiddleware

        mock_timer_stop.side_effect = TimerError
        key = 'timer1'
        metric = RequestMetricsStore.for_request(self.request)

        with self.assertLogs(level='WARNING') as cm:
            metric.stop(key)
        self.assertIn(f"timer {key} stopped incorrectly. Time not recorded.", cm.output[0])

    @patch.object(Timer, 'stop')
    def test_stop_fail(self, mock_timer_stop):
        RequestMetricsStore.set_in_request(r=self.request) # As set by TelemetryMiddleware

        mock_timer_stop.side_effect = TimerError
        key = 'timer1'
        metric = RequestMetricsStore.for_request(self.request)
        with self.assertLogs(level='WARNING') as cm:
            metric.stop(key)
        self.assertIn(f"timer {key} stopped incorrectly. Time not recorded.", cm.output[0])

    def test_increment_counter_and_json(self):
        RequestMetricsStore.set_in_request(r=self.request) # As set by TelemetryMiddleware

        key = 'key1'
        metric = RequestMetricsStore.for_request(self.request)
        metric.increment_counter(key)
        metric.times[key] = 1.0
        expected_json = {"counter": {key: 1}, "timesMs": {key: 1.0}}
        self.assertEqual(expected_json, metric.json())


class TestTelemetryMiddleware(unittest.TestCase):

    def setUp(self):
        self.app = Starlette()
        self.app.add_middleware(TelemetryMiddleware)

        @self.app.route("/", methods=["GET"])
        def test_endpoint(request):
            return JSONResponse({"data": "test"})
        
        self.client = TestClient(self.app)
        self.scope = {'type': 'http'}
        self.request = Request(self.scope)

    def test_telemetry_disabled(self):
        response = self.client.get("/")
        self.assertNotIn("telemetry", response.json())

        response = self.client.get("/?telemetry=false")
        self.assertNotIn("telemetry",    response.json())

    def test_telemetry_enabled(self):
        response = self.client.get("/?telemetry=true")
        self.assertIn("telemetry", response.json())

    @unittest.skip("Error running in GH Actions")
    def test_counter_usage(self):
        @self.app.route("/test", methods=["GET"])
        def test_endpoint(request):
            m = RequestMetricsStore.for_request()
            m.increment_counter("key")
            return JSONResponse({"data": "test"})
        
        response = self.client.get("/test?telemetry=true")
        self.assertIn("telemetry", response.json())
        self.assertIn("counter", response.json()["telemetry"])
        self.assertEqual(response.json()["telemetry"], {
            "counter": {"key": 1.0}
        })

    @unittest.skip("Error running in GH Actions")
    def test_timing_usage(self):
        @self.app.route("/test", methods=["GET"])
        def test_endpoint(request):
            m = RequestMetricsStore.for_request()
            m.start("key")
            m.stop("key")
            return JSONResponse({"data": "test"})
        
        response = self.client.get("/test?telemetry=true")
        self.assertIn("telemetry", response.json())
        self.assertIn("timesMs", response.json()["telemetry"])
        self.assertIn("key", response.json()["telemetry"]["timesMs"])

    @unittest.skip("Error running in GH Actions")
    def test_with_timing_usage(self):
        @self.app.route("/test", methods=["GET"])
        def test_endpoint(request):
            m = RequestMetricsStore.for_request()
            with m.time("key"):
                pass
            return JSONResponse({"data": "test"})
        
        response = self.client.get("/test?telemetry=true")
        self.assertIn("telemetry", response.json())
        self.assertIn("timesMs", response.json()["telemetry"])
        self.assertIn("key", response.json()["telemetry"]["timesMs"])

    def test_custom_telemetry_flag(self):
        middleware = [
            Middleware(TelemetryMiddleware, telemetery_flag="custom_telemetry")
        ]
        app = Starlette(middleware=middleware)

        @app.route("/", methods=["GET"])
        def test_endpoint(request):
            return JSONResponse({"data": "test"})
        
        client = TestClient(app)
        response = client.get("/?custom_telemetry=true")
        self.assertIn("telemetry", response.json())

        response = client.get("/?custom_telemetry=false")
        self.assertNotIn("telemetry", response.json())

        response = client.get("/?telemetry=true")
        self.assertNotIn("telemetry", response.json())

    @patch('starlette.testclient.TestClient.send')
    async def test_dispatch_no_telemetry(self, mock_send):
        # Mock the send function to return a mock response
        mock_send.return_value = Response()

        async def call_next(request):
            return Response()

        response = await self.middleware.dispatch(self.request, call_next)
        self.assertIsInstance(response, Response)
        self.assertNotIn("telemetry", response.body.decode())

    @patch('starlette.testclient.TestClient.send')
    async def test_dispatch_telemetry_not_dict(self, mock_send):
        mock_send.return_value = Response()

        async def call_next(request):
            return Response(content=json.dumps(["not", "a", "dict"]))

        self.scope = {'type': 'http', 'query_string': b'telemetry=true'}
        self.request = Request(self.scope)
        response = await self.middleware.dispatch(self.request, call_next)
        self.assertIsInstance(response, Response)
        self.assertNotIn("telemetry", response.body.decode())

    @patch('starlette.testclient.TestClient.send')
    async def test_dispatch_telemetry_dict(self, mock_send):
        self.scope = {'type': 'http', 'query_string': b'telemetry=true'}
        self.request = Request(self.scope)

        # Mock the send function to return a mock response
        mock_send.return_value = Response()

        async def call_next(request):
            return Response(content=json.dumps({"data": "value"}))

        response = await self.middleware.dispatch(self.request, call_next)
        self.assertIsInstance(response, Response)
        self.assertIn("telemetry", response.body.decode())

    def test_telemetry_enabled_for_request(self):
        self.scope = {'type': 'http', 'query_string': b'telemetry=true'}
        self.request = Request(self.scope)
        self.assertTrue(TelemetryMiddleware(self.app).telemetry_enabled_for_request(self.request))
        
        self.scope = {'type': 'http', 'query_string': b'telemetry=false'}
        self.request = Request(self.scope)
        self.assertFalse(TelemetryMiddleware(self.app).telemetry_enabled_for_request(self.request))
