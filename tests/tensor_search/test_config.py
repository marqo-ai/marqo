import unittest
from unittest import mock
from marqo import config
import torch
from tests.marqo_test import MarqoTestCase
from marqo.tensor_search import enums


@unittest.skip
class TestConfig(MarqoTestCase):

    def setUp(self) -> None:
        self.endpoint = self.authorized_url

    def test_set_url_localhost(self):
        def run():
            c = config.Config(url="https://localhost:9200")
            assert not c.cluster_is_remote
            return True
        assert run()

    def test_set_url_0000(self):
        def run():
            c = config.Config(url="https://0.0.0.0:9200")
            assert not c.cluster_is_remote
            return True
        assert run()

    def test_set_url_127001(self):
        def run():
            c = config.Config(url="https://127.0.0.1:9200")
            assert not c.cluster_is_remote
            return True
        assert run()

    def test_device_for_clip(self):
        assert str(enums.Device.cpu) == "cpu"


class TestConfigBackend(MarqoTestCase):

    def setUp(self) -> None:
        self.endpoint = self.authorized_url

    class CustomSearchDb:
        opensearch = "opensearch"
        elasticsearch = "elasticsearch"

    def test_init_default_backend(self):
        c = config.Config(url=self.endpoint)
        assert c.backend == enums.SearchDb.opensearch

    def test_init_custom_backend(self):
        c = config.Config(url=self.endpoint, backend=self.CustomSearchDb.elasticsearch)
        assert c.backend == "elasticsearch"

    def test_init_custom_backend_as_string(self):
        c = config.Config(url=self.endpoint, backend="elasticsearch")
        assert c.backend == "elasticsearch"
