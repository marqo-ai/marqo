import unittest

from marqo.errors import IndexNotFoundError
from marqo.tensor_search import tensor_search
from marqo.tensor_search.telemetry import RequestMetricsStore
from marqo.tensor_search.utils import construct_authorized_url
from marqo import config
from unittest.mock import patch, Mock


class MarqoTestCase(unittest.TestCase):

    @classmethod
    def configure_request_metrics(cls):
        """Mock RequestMetricsStore to avoid complications with not having TelemetryMiddleware configuring metrics.
        """
        cls.mock_request = Mock()
        cls.patcher = patch('marqo.tensor_search.telemetry.RequestMetricsStore._get_request')
        cls.mock_get_request = cls.patcher.start()
        cls.mock_get_request.return_value = cls.mock_request
        RequestMetricsStore.set_in_request(cls.mock_request)

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

        # Delete any test indices my-test-index-i (up to 5)
        for ix_name in [f"my-test-index-{i}" for i in range(1, 6)]:
            try:
                tensor_search.delete_index(config=cls.config, index_name=ix_name)
            except IndexNotFoundError:
                pass

    @classmethod
    def setUpClass(cls) -> None:
        cls.configure_request_metrics()

        # Set up the Marqo root dir (for use in model caches)
        local_opensearch_settings = {
            "url": 'https://localhost:9200',
            "main_user": "admin",
            "main_password": "admin"
        }
        cls.client_settings = local_opensearch_settings
        cls.authorized_url = construct_authorized_url(
            url_base=cls.client_settings["url"],
            username=cls.client_settings["main_user"],
            password=cls.client_settings["main_password"]
        )
        cls.config = config.Config(url=cls.authorized_url)
