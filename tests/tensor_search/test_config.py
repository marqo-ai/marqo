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
            assert c.indexing_device == enums.Devices.cpu
            assert c.search_device == enums.Devices.cpu
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
        @mock.patch("urllib3.disable_warnings")
        def run(mock_dis_warnings):
            c = config.Config(url="https://localhost:9200")
            assert not c.cluster_is_remote
            mock_dis_warnings.assert_called()
            return True
        assert run()

    def test_set_url_0000(self):
        @mock.patch("urllib3.disable_warnings")
        def run(mock_dis_warnings):
            c = config.Config(url="https://0.0.0.0:9200")
            assert not c.cluster_is_remote
            mock_dis_warnings.assert_called()
            return True
        assert run()

    def test_set_url_127001(self):
        @mock.patch("urllib3.disable_warnings")
        def run(mock_dis_warnings):
            c = config.Config(url="https://127.0.0.1:9200")
            assert not c.cluster_is_remote
            mock_dis_warnings.assert_called()
            return True
        assert run()

    def test_set_url_remote(self):
        @mock.patch("urllib3.disable_warnings")
        @mock.patch("warnings.resetwarnings")
        def run(mock_reset_warnings, mock_dis_warnings):
            c = config.Config(url="https://some-cluster-somewhere:9200")
            assert c.cluster_is_remote
            mock_dis_warnings.assert_not_called()
            mock_reset_warnings.assert_called()
            return True
        assert run()

    def test_url_is_s2search(self):
        c = config.Config(url="https://s2search.io/abdcde:9200")
        assert c.cluster_is_s2search

    def test_url_is_not_s2search(self):
        c = config.Config(url="https://som_random_cluster/abdcde:9200")
        assert not c.cluster_is_s2search
