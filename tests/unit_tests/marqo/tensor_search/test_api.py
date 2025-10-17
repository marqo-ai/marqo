import unittest
from unittest.mock import Mock, patch

from starlette.testclient import TestClient

from marqo.tensor_search import api


class TestApiInitialisation(unittest.TestCase):
    @patch("marqo.tensor_search.api.bootstrap_otel")
    def test_lifespan_integration_bootstrap_and_shutdown_otel(
        self, mock_bootstrap_otel
    ):
        mock_otel_shutdown_hook = Mock()
        mock_bootstrap_otel.return_value = mock_otel_shutdown_hook

        # Use FastAPI TestClient to simulate making a request to the app
        with TestClient(api.app) as _:
            # Ensure the shutdown hook was called and Zookeeper stop method was triggered
            mock_bootstrap_otel.assert_called_once_with(
                api.app, service_name="marqo-api"
            )

        mock_otel_shutdown_hook.assert_called_once()
