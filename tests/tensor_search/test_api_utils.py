import pydantic
from marqo.tensor_search.models.add_docs_objects import ModelAuth, AddDocsParams, AddDocsBodyParamsOld, \
    AddDocsBodyParamsNew
from marqo.tensor_search.web.api_utils import add_docs_params_orchestrator
from marqo.tensor_search.models.private_models import S3Auth
import urllib.parse
from marqo.tensor_search.web import api_utils
from marqo.errors import InvalidArgError, InternalError, BadRequestError
from tests.marqo_test import MarqoTestCase
import unittest


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
            ("http://king_user:mysecretpw@unusual.com/happy@chappy:9200",
             "http://king_user:mysecretpw@unusual.com/happy@chappy:9200"),
            ("http://unusual.com/happy@chappy:9200", "http://admin:admin@unusual.com/happy@chappy:9200"),
            (
            "http://www.unusual.com/happy@@@@#chappy:9200", "http://admin:admin@www.unusual.com/happy@@@@#chappy:9200"),
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


class TestAddDocsParamsOchestrator(unittest.TestCase):
    def test_add_docs_params_orchestrator_new(self):
        # Set up the arguments for the function
        index_name = "test-index"
        body = AddDocsBodyParamsNew(documents=[{"test": "doc"}],
                                    nonTensorFields=["field1"],
                                    useExistingTensors=True,
                                    imageDownloadHeaders={"header1": "value1"},
                                    modelAuth=ModelAuth(s3=S3Auth(aws_secret_access_key="test", aws_access_key_id="test")),
                                    mappings={"map1": "value1"})
        device = "test-device"
        auto_refresh = True

        # Query parameters should be parsed as default values
        non_tensor_fields = []
        use_existing_tensors = False
        image_download_headers = dict()
        model_auth = None
        mappings = dict()

        # Call the function with the arguments
        result = add_docs_params_orchestrator(index_name, body, device, auto_refresh, non_tensor_fields, mappings,
                                              model_auth, image_download_headers, use_existing_tensors)

        # Assert that the result is as expected
        assert isinstance(result, AddDocsParams)
        assert result.index_name == "test-index"
        assert result.docs == body.documents
        assert result.device == "test-device"
        assert result.non_tensor_fields == ["field1"]
        assert result.use_existing_tensors == True
        assert result.docs == [{"test": "doc"}]
        assert result.image_download_headers == {"header1": "value1"}

    def test_add_docs_params_orchestrator_old(self):
        # Set up the arguments for the function
        index_name = "test-index"
        model_auth = ModelAuth(s3=S3Auth(aws_secret_access_key="test", aws_access_key_id="test"))

        body = AddDocsBodyParamsOld(__root__= [{"test": "doc"}])

        device = "test-device"
        non_tensor_fields = ["field1"]
        use_existing_tensors = True
        image_download_headers = {"header1": "value1"}
        model_auth = model_auth
        mappings = {"map1": "value1"}
        auto_refresh = True

        # Call the function with the arguments
        result = add_docs_params_orchestrator(index_name, body, device, auto_refresh, non_tensor_fields, mappings,
                                              model_auth, image_download_headers, use_existing_tensors)

        # Assert that the result is as expected
        assert isinstance(result, AddDocsParams)
        assert result.index_name == "test-index"
        assert result.docs == body.__root__
        assert result.device == "test-device"
        assert result.non_tensor_fields == ["field1"]
        assert result.use_existing_tensors == True
        assert result.docs == [{"test": "doc"}]
        assert result.image_download_headers == {"header1": "value1"}

    def test_add_docs_params_orchestrator_error(self):
        # Test the case where the function should raise an error due to invalid input
        body = "invalid body type"  # Not an instance of AddDocsBodyParamsNew or AddDocsBodyParamsOld

        index_name = "test-index"
        model_auth = ModelAuth(s3=S3Auth(aws_secret_access_key="test", aws_access_key_id="test"))

        device = "test-device"
        non_tensor_fields = ["field1"]
        use_existing_tensors = True
        image_download_headers = {"header1": "value1"}
        model_auth = model_auth
        mappings = {"map1": "value1"}
        auto_refresh = True

        # Use pytest.raises to check for the error
        try:
           _ = add_docs_params_orchestrator(index_name, body, device, auto_refresh, non_tensor_fields, mappings,
                                            model_auth, image_download_headers, use_existing_tensors)
        except BadRequestError as e:
            self.assertIn("Invalid request body", str(e))

    def test_add_docs_params_orchestrator_depreciated_query_parameters_error(self):
        # Test the case where the function should raise an error due to depreciated query parameters
        index_name = "test-index"
        model_auth = ModelAuth(s3=S3Auth(aws_secret_access_key="test", aws_access_key_id="test"))
        device = "test-device"
        auto_refresh = True
        body = AddDocsBodyParamsNew(documents=[{"test": "doc"}],
                                    nonTensorFields=["field1"],
                                    useExistingTensors=True,
                                    imageDownloadHeaders={"header1": "value1"},
                                    modelAuth=ModelAuth(s3=S3Auth(aws_secret_access_key="test", aws_access_key_id="test")),
                                    mappings={"map1": "value1"})

        params = {"non_tensor_fields": ["what"], "use_existing_tensors": True,
                  "image_download_headers": {"header2": "value2"}, "model_auth": model_auth,
                  "mappings": {"map2": "value2"}}

        for param, value in params.items():
            kwargs = {key: None for key in params.keys()}
            kwargs[param] = value
            try:
                add_docs_params_orchestrator(index_name, body, device, auto_refresh=auto_refresh,
                                             query_parameters=kwargs, **kwargs)
            except BadRequestError as e:
                self.assertIn("Marqo is not accepting any of the following parameters in the query string", str(e))
