from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from marqo import exceptions as base_exceptions
from marqo.api import exceptions as api_exceptions
from marqo.api.route import MarqoCustomRoute
from tests.marqo_test import MarqoTestCase

app = FastAPI()
app.router.route_class = MarqoCustomRoute


@app.get("/test-route")
async def test_route():
    raise ValueError("Test Error for MarqoCustomRoute")


@app.get("/raise-api-exception")
async def test_raise_api_exception():
    raise api_exceptions.MarqoWebError("Test API exceptions for MarqoCustomRoute")


@app.get("/raise-base-exception")
async def test_raise_base_exception():
    raise base_exceptions.MarqoError("Test Base exceptions for MarqoCustomRoute")


@app.get("/normal-route")
async def test_route():
    return {"message": "Hello, World!"}

client = TestClient(app)


class TestMarqoCustomRoute(MarqoTestCase):
    def test_marqo_custom_route_logs_error(self):
        with patch('marqo.api.route.logger.error') as mock_logger_error:
            with self.assertRaises(ValueError):
                response = client.get("/test-route")
            mock_logger_error.assert_called_once()
            self.assertIn("Test Error for MarqoCustomRoute", str(mock_logger_error.call_args))

    def test_marqo_custom_route_logs_api_exception(self):
        expected_error = api_exceptions.MarqoWebError("Test API exceptions for MarqoCustomRoute")
        with patch('marqo.api.route.logger.error') as mock_logger_error:
            with self.assertRaises(api_exceptions.MarqoWebError):
                response = client.get("/raise-api-exception")
            mock_logger_error.assert_called_once_with(str(expected_error), exc_info=True)
            self.assertIn("Test API exceptions for MarqoCustomRoute", str(mock_logger_error.call_args))

    def test_marqo_custom_route_base_exception(self):
        expected_error = base_exceptions.MarqoError("Test Base exceptions for MarqoCustomRoute")
        with patch('marqo.api.route.logger.error') as mock_logger_error:
            with self.assertRaises(base_exceptions.MarqoError):
                response = client.get("/raise-base-exception")
            mock_logger_error.assert_called_once_with(str(expected_error), exc_info=True)
            self.assertIn("Test Base exceptions for MarqoCustomRoute", str(mock_logger_error.call_args))

    def test_normal_route(self):
        with patch('marqo.api.route.logger.error') as mock_logger_error:
            response = client.get("/normal-route")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"message": "Hello, World!"})
            mock_logger_error.assert_not_called()