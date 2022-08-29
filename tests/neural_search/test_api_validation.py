import requests
from marqo.neural_search import enums, backend
from marqo.neural_search import neural_search
from marqo.neural_search.web import api_validation
from marqo.errors import InvalidArgError
from tests.marqo_test import MarqoTestCase


class TestApiValidation(MarqoTestCase):

    def test_validate_api_device_good(self):
        for good in ["cpu", "cuda", "CPU", "CUDA2", "cuda1234", "cpu1"]:
            assert good == api_validation.validate_api_device(good)

    def test_validate_api_device_bad(self):
        for bad in [dict(), set(), 123, "CUDA:1", "JKJKNN","cpu:3", "cuda:3"]:
            try:
                api_validation.validate_api_device(bad)
                print(bad)
                raise AssertionError
            except InvalidArgError:
                pass
