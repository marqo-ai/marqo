import unittest
from unittest.mock import patch

from fastapi import Request

from starlette.applications import Starlette
from starlette.testclient import TestClient
from starlette.responses import JSONResponse
from starlette.middleware import Middleware

from marqo.tensor_search.telemetry import RequestMetrics, TelemetryMiddleware, Timer, TimerError


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
        self.assertEqual(elapsed_time, 1.0)
        self.assertIsNone(self.timer.start_time)
        
        # Test warning if stopping without starting
        with self.assertLogs(level='WARNING') as cm:
            with self.assertRaises(TimerError):
                self.timer.stop()
        self.assertIn("'.stop()' called on unstarted timer.", cm.output[0])

    @patch('time.perf_counter')
    def test_timer_restart(self, mock_time):
        # Simulate the time progression
        mock_time.side_effect = [0.0, 1.0, 2.0, 4.0]
        
        # Test timer start, stop, and restart
        self.timer.start()
        elapsed_time = self.timer.stop()
        self.assertEqual(elapsed_time, 1.0)
        
        self.timer.start()
        elapsed_time = self.timer.stop()
        self.assertEqual(elapsed_time, 2.0)


class TestRequestMetrics(unittest.TestCase):

    def setUp(self):
        self.request_metrics = RequestMetrics()
        self.request = Request(scope={"type": "http"})

    def tearDown(self) -> None:
        RequestMetrics.clear_metrics_for(self.request)

    def test_set_in_request(self):
        self.assertEqual({}, RequestMetrics.METRIC_STORES)

        RequestMetrics.set_in_request(self.request)
        RequestMetrics.for_request(self.request)

    def test_clear_metrics_for(self):
        RequestMetrics.set_in_request(self.request)
        self.assertIsNotNone(RequestMetrics.for_request(self.request))

        RequestMetrics.clear_metrics_for(self.request)
        self.assertEqual({}, RequestMetrics.METRIC_STORES)

    def test_for_request(self):
        RequestMetrics.set_in_request(self.request)
        self.assertIn(self.request, RequestMetrics.METRIC_STORES)

    def test_increment_counter(self):
        key = 'counter1'
        self.request_metrics.increment_counter(key)
        self.assertEqual(
            self.request_metrics.json(),
            {
                "counter": {key: 1.0},
                "times": {}
        })
        self.assertEqual({key: 1}, self.request_metrics.counter)

    @patch.object(Timer, 'start')
    @patch.object(Timer, 'stop')
    def test_time(self, mock_timer_stop, mock_timer_start):
        mock_timer_stop.return_value = 1.0
        key = 'timer1'

        with self.request_metrics.time(key):
            pass
        
        self.assertEqual(
            self.request_metrics.json(),
            {
                "counter": {},
                "times": {key: 1.0}
        })
        mock_timer_start.assert_called_once()
        mock_timer_stop.assert_called_once()

    @patch.object(Timer, 'start')
    def test_start(self, mock_timer_start):
        self.request_metrics.start('timer1')
        mock_timer_start.assert_called_once()

    @patch.object(Timer, 'stop')
    def test_stop_success(self, mock_timer_stop):
        mock_timer_stop.return_value = 1.0
        key = 'timer1'
        elapsed_time = self.request_metrics.stop(key)
        self.assertEqual(
            self.request_metrics.json(),
            {
                "counter": {},
                "times": {key: 1.0}
        })
        self.assertEqual(1.0, elapsed_time)
        mock_timer_stop.assert_called_once()

    @patch.object(Timer, 'stop')
    def test_stop_fail(self, mock_timer_stop):
        mock_timer_stop.side_effect = TimerError
        key = 'timer1'

        with self.assertLogs(level='WARNING') as cm:
            self.request_metrics.stop(key)
        self.assertIn("timer timer1 stopped incorrectly. Time not recorded.", cm.output[0])

    def test_json(self):
        key = 'key1'
        self.request_metrics.increment_counter(key)
        self.request_metrics.times[key] = 1.0
        expected_json = {"counter": {key: 1}, "times": {key: 1.0}}
        self.assertEqual(expected_json, self.request_metrics.json())


class TestTelemetryMiddleware(unittest.TestCase):

    def setUp(self):
        middleware = [
            Middleware(TelemetryMiddleware)
        ]
        self.app = Starlette(middleware=middleware)

        @self.app.route("/", methods=["GET"])
        def test_endpoint(request):
            return JSONResponse({"data": "test"})
        
        self.client = TestClient(self.app)

    def test_telemetry_disabled(self):
        response = self.client.get("/")
        self.assertNotIn("telemetry", response.json())

        response = self.client.get("/?telemetry=false")
        self.assertNotIn("telemetry", response.json())

    def test_telemetry_enabled(self):
        response = self.client.get("/?telemetry=true")
        self.assertIn("telemetry", response.json())

    def test_counter_usage(self):
        @self.app.route("/test", methods=["GET"])
        def test_endpoint(request):
            m = RequestMetrics.for_request()
            m.increment_counter("key")
            return JSONResponse({"data": "test"})
        
        response = self.client.get("/test?telemetry=true")
        self.assertIn("telemetry", response.json())
        self.assertEqual(response.json()["telemetry"], {
            "counter": {"key": 1.0},
            "times": {}
        })

    def test_timing_usage(self):
        @self.app.route("/test", methods=["GET"])
        def test_endpoint(request):
            m = RequestMetrics.for_request()
            m.start("key")
            m.stop("key")
            return JSONResponse({"data": "test"})
        
        response = self.client.get("/test?telemetry=true")
        self.assertIn("telemetry", response.json())
        self.assertIn("key", response.json()["telemetry"]["times"])


    def test_with_timing_usage(self):
        @self.app.route("/test", methods=["GET"])
        def test_endpoint(request):
            m = RequestMetrics.for_request()
            with m.time("key"):
                pass
            return JSONResponse({"data": "test"})
        
        response = self.client.get("/test?telemetry=true")
        self.assertIn("telemetry", response.json())
        self.assertIn("key", response.json()["telemetry"]["times"])

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