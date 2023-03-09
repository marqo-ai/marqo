import unittest
from unittest import mock
from marqo import config
import torch
from tests.marqo_test import MarqoTestCase
from marqo.tensor_search import enums


class TestConfig(MarqoTestCase):

    def setUp(self) -> None:
        self.endpoint = self.authorized_url

    def test_init_custom_devices(self):
        c = config.Config(url=self.endpoint,indexing_device="cuda:3", search_device="cuda:4")
        assert c.indexing_device == "cuda:3"
        assert c.search_device == "cuda:4"

    def test_init_infer_gpu_device(self):
        mock_torch_cuda = mock.MagicMock()
        mock_torch_cuda.is_available.return_value = True

        @mock.patch("torch.cuda", mock_torch_cuda)
        def run():
            c = config.Config(url=self.endpoint, indexing_device='cuda' if torch.cuda.is_available() else 'cpu', 
                                    search_device='cuda' if torch.cuda.is_available() else 'cpu')
            assert c.indexing_device == enums.Device.cuda, f"{enums.Device.cuda} {c.indexing_device}"
            assert c.search_device == enums.Device.cuda
            return True
        assert run()

    def test_init_infer_cpu_device(self):
        mock_torch_cuda = mock.MagicMock()
        mock_torch_cuda.is_available.return_value = False

        @mock.patch("torch.cuda", mock_torch_cuda)
        def run():
            c = config.Config(url=self.endpoint)
            assert c.indexing_device == enums.Device.cpu
            assert c.search_device == enums.Device.cpu
            return True
        assert run()

    def test_init_override_inferred_device(self):
        mock_torch_cuda = mock.MagicMock()
        mock_torch_cuda.is_available.return_value = True

        @mock.patch("torch.cuda", mock_torch_cuda)
        def run():
            c = config.Config(url=self.endpoint, indexing_device="cuda:3", search_device="cuda:4")
            assert c.indexing_device == "cuda:3"
            assert c.search_device == "cuda:4"
            return True
        assert run()

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
