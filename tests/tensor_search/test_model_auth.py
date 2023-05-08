from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from marqo.tensor_search.models.private_models import S3Auth, ModelAuth, HfAuth
from marqo.errors import InvalidArgError, IndexNotFoundError
from tests.marqo_test import MarqoTestCase
from marqo.s2_inference.model_downloading.from_s3 import get_s3_model_absolute_cache_path
from marqo.tensor_search.models.external_apis.s3 import S3Location
from unittest import mock
import unittest
import os


class TestModelAuth(MarqoTestCase):

    def setUp(self) -> None:
        self.endpoint = self.authorized_url
        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "test-model-auth-index-1"
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass

    @staticmethod
    def _delete_file(file_path):
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass

    @staticmethod
    def _get_base_index_settings():
        return {
            "index_defaults": {
                "treat_urls_and_pointers_as_images": True,
                "model": 'my_model',
                "normalize_embeddings": True,
                # notice model properties aren't here. Each test has to add it
            }
        }

    def tearDown(self) -> None:
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass

    def test_model_auth_s3(self):
        """Simulates downloading a model from a private and using it in an
        add docs call
        """

        s3_object_key = 'path/to/your/secret_model.pt'
        s3_bucket = 'your-bucket-name'

        model_abs_path = get_s3_model_absolute_cache_path(
            S3Location(
                Key=s3_object_key,
                Bucket=s3_bucket
        ))
        self._delete_file(model_abs_path)

        model_properties = {
            "name": "ViT-B/32",
            "dimensions": 512,
            "model_location": {
                "s3": {
                    "Bucket": s3_bucket,
                    "Key": s3_object_key,
                },
                "auth_required": True
            },
            "type": "open_clip",
        }
        s3_settings = self._get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)

        fake_access_key_id = '12345'
        fake_secret_key = 'this-is-a-secret'
        public_model_url = "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt"

        # Create a mock Boto3 client
        mock_s3_client = mock.MagicMock()

        # Mock the generate_presigned_url method of the mock Boto3 client with a real OpenCLIP model, so that
        # the rest of the logic works.
        mock_s3_client.generate_presigned_url.return_value = public_model_url

        # file should not yet exist:
        assert not os.path.isfile(model_abs_path)

        with unittest.mock.patch('boto3.client', return_value=mock_s3_client)  as mock_boto3_client:
            # Call the function that uses the generate_presigned_url method
            tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1, auto_refresh=True, docs=[{'a': 'b'}],
                model_auth=ModelAuth(s3=S3Auth(aws_access_key_id=fake_access_key_id, aws_secret_access_key=fake_secret_key))
            ))

        assert os.path.isfile(model_abs_path)

        mock_s3_client.generate_presigned_url.assert_called_with(
            'get_object',
            Params={'Bucket': 'your-bucket-name', 'Key': s3_object_key}
        )
        mock_boto3_client.assert_called_once_with(
            's3',
            aws_access_key_id=fake_access_key_id,
            aws_secret_access_key=fake_secret_key,
            aws_session_token=None
        )
        self._delete_file(model_abs_path)

    def test_model_auth_hf(self):
        model_properties = {
            "name": "ViT-B/32",
            "dimensions": 512,
            "model_location": {
                "s3": {
                    "repo_id": "Marqo/test-private",
                    "filename": "dummy_model.pt",
                },
                "auth_required": True
            },
            "type": "clip",
        }
        s3_settings = self._get_base_index_settings()
        s3_settings['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)
        # TODO: mock call to HF
        raise NotImplementedError

    def test_model_loads_from_search(self):
        """The other ones load from add_docs, we have to make sure it works for
         search"""

    def test_model_loads_from_all_add_docs_derivatives(self):
        """Does it work from add_docs, add_docs orchestrator and add_documents_mp?
        """

    def test_model_loads_from_multi_search(self):
        pass

    def test_model_loads_from_multimodal_combination(self):
        pass

    def test_no_creds_error(self):
        """in s3, if there aren't creds"""

    def test_bad_creds_error(self):
        """in s3, hf if creds aren't valid. Ensure a helpful error"""

    def test_doesnt_redownload_s3(self):
        """We also need to ensure that it doesn't redownload from add docs to search
        and vice vers """

    def test_downloaded_to_correct_dir(self):
        """"""

    def test_public_s3_no_auth(self):
        """"""

    def test_public_hf_no_auth(self):
        """"""

    def test_open_clip_reg_clip(self):
        """both normal and open clip"""

    ################## TESTS FOR VECTORISE PARAMS

    def test_as_dict_discards_nones_expected_dict(self):
        """assert the dict is structured as expected"""

        """Assert __pydantic_initialised__ is not in the dict
        
        test both k and value staring with __ 
        """








