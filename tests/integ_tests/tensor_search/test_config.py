import unittest
from unittest import mock
from marqo import config
import torch
from integ_tests.marqo_test import MarqoTestCase
from marqo.tensor_search import enums
from marqo.tensor_search.api import generate_config
import os
from unittest import mock
from marqo.tensor_search.enums import EnvVars

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




@unittest.skip
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


class TestGenerateConfig(MarqoTestCase):

    def test_configWithoutZookeeperHostsBeingSet(self):
        """Test that the config is generated correctly when ZOOKEEPER_HOSTS is not set or is an empty string."""
        environment_variable_test_cases = [
            {"ZOOKEEPER_HOSTS": ""},  # Empty string
            dict()  # Empty dict, unset
        ]
        for env in environment_variable_test_cases:
            with self.subTest(env):
                with mock.patch.dict(os.environ, env):
                    c = generate_config()
                    self.assertIsNone(c._zookeeper_client)

    def test_configWithZookeeperHostsBeingSet(self):
        """Test that the config is generated correctly when ZOOKEEPER_HOSTS is set to a value."""
        env = {"ZOOKEEPER_HOSTS": "a.fake.url"}
        with mock.patch.dict(os.environ, env):
            with mock.patch("marqo.config.Config._connect_to_zookeeper") as mock_connect_to_zookeeper:
                c = generate_config()
                mock_connect_to_zookeeper.assert_called_once()
                self.assertIsNotNone(c._zookeeper_client)