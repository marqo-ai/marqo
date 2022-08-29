import requests
from marqo.neural_search import enums, backend
from marqo.neural_search import neural_search
from marqo.neural_search.web import api_utils
from marqo.errors import InvalidArgError
from tests.marqo_test import MarqoTestCase


class TestApiUtils(MarqoTestCase):

    def test_validate_api_device_good(self):
        for given, expected in [("cpu", "cpu"), ("cuda", "cuda"),
                                ("CPU", "cpu"), ("CUDA2", "cuda:2"),
                                ("cuda1234", "cuda:1234"), ("cpu1", "cpu:1")]:
            assert expected == api_utils.translate_api_device(given)

    def test_validate_api_device_bad(self):
        for bad in ["avr", "123"]:
            try:
                api_utils.translate_api_device(bad)
                raise AssertionError
            except InvalidArgError:
                pass
