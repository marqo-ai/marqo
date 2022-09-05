import requests
from marqo.tensor_search import enums, backend
from marqo.tensor_search import tensor_search
from marqo.tensor_search.web import api_utils
from marqo.errors import InvalidArgError, InternalError
from tests.marqo_test import MarqoTestCase


class TestApiUtils(MarqoTestCase):

    def test_translate_api_device_good(self):
        for given, expected in [("cpu", "cpu"), ("cuda", "cuda"),
                                ("CPU", "cpu"), ("CUDA2", "cuda:2"),
                                ("cuda1234", "cuda:1234"), ("cpu1", "cpu:1"),
                                (None, None)]:
            assert expected == api_utils.translate_api_device(given)

    def test_translate_api_device_bad(self):
        for bad in ["avr", "123"]:
            try:
                api_utils.translate_api_device(bad)
                raise AssertionError
            except InvalidArgError:
                pass

    def test_generate_config(self):
        for opensearch_url, authorized_url in [
                ("http://admin:admin@localhost:9200", "http://admin:admin@localhost:9200"),
                ("http://localhost:9200", "http://admin:admin@localhost:9200"),
                ("https://admin:admin@localhost:9200", "https://admin:admin@localhost:9200"),
                ("https://localhost:9200", "https://admin:admin@localhost:9200"),
                ("http://king_user:mysecretpw@unusual.com/happy@chappy:9200", "http://king_user:mysecretpw@unusual.com/happy@chappy:9200"),
                ("http://unusual.com/happy@chappy:9200", "http://admin:admin@unusual.com/happy@chappy:9200"),
                ("http://www.unusual.com/happy@@@@#chappy:9200", "http://admin:admin@www.unusual.com/happy@@@@#chappy:9200"),
                ("://", "://admin:admin@")
                ]:
            c = api_utils.upconstruct_authorized_url(opensearch_url=opensearch_url)
            assert authorized_url == c.url

    def test_generate_config_bad_url(self):
        for opensearch_url in ["www.google.com", "http:/mywebsite", "yahoo"]:
            try:
                c = api_utils.upconstruct_authorized_url(opensearch_url=opensearch_url)
                raise AssertionError
            except InternalError:
                pass