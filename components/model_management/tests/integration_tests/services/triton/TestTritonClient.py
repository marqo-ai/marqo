from model_management.services.triton import triton_client
from model_management.services.errors import TritonModelLoadError
from unittest import TestCase


class TestTritonClient(TestCase):
    def test_load_unload_model_success(self):
        client = triton_client.TritonClient("http://localhost:8000")
        with self.assertRaises(TritonModelLoadError):
            client.load_model("void-model")
