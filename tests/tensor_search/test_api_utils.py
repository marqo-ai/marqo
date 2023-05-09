import pydantic
from marqo.tensor_search.models.add_docs_objects import ModelAuth
from marqo.tensor_search.models.private_models import S3Auth
import urllib.parse
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
            assert authorized_url == c

    def test_generate_config_bad_url(self):
        for opensearch_url in ["www.google.com", "http:/mywebsite", "yahoo"]:
            try:
                c = api_utils.upconstruct_authorized_url(opensearch_url=opensearch_url)
                raise AssertionError
            except InternalError:
                pass
            
class TestDecodeQueryStringModelAuth(MarqoTestCase):

    def test_decode_query_string_model_auth_none(self):
        result = api_utils.decode_query_string_model_auth()
        self.assertIsNone(result)

    def test_decode_query_string_model_auth_empty_string(self):
        result = api_utils.decode_query_string_model_auth("")
        self.assertIsNone(result)

    def test_decode_query_string_model_auth_valid(self):
        model_auth_obj = ModelAuth(s3=S3Auth(
            aws_access_key_id='some_acc_id', aws_secret_access_key='some_sece_key'))
        model_auth_str = model_auth_obj.json()
        model_auth_url_encoded = urllib.parse.quote_plus(model_auth_str)

        result = api_utils.decode_query_string_model_auth(model_auth_url_encoded)

        self.assertIsInstance(result, ModelAuth)
        self.assertEqual(result.s3.aws_access_key_id, 'some_acc_id')
        self.assertEqual(result.s3.aws_secret_access_key, 'some_sece_key')
        self.assertEqual(result.hf, None)

    def test_decode_query_string_model_auth_invalid(self):
        with self.assertRaises(pydantic.ValidationError):
            api_utils.decode_query_string_model_auth("invalid_url_encoded_string")