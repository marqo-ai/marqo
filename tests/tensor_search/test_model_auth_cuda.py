"""todos: get a public HF-based ViT model so that we can use it for mocks and tests

multiprocessing should be tested manually -problem with mocking (deadlock esque)
"""
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from marqo.tensor_search.models.private_models import S3Auth, ModelAuth, HfAuth
from marqo.errors import InvalidArgError, IndexNotFoundError, BadRequestError
from tests.marqo_test import MarqoTestCase
from marqo.s2_inference.model_downloading.from_s3 import get_s3_model_absolute_cache_path
from marqo.tensor_search.models.external_apis.s3 import S3Location
from unittest import mock
from tests.tensor_search.test_model_auth import _delete_file, _get_base_index_settings
import unittest
import os
import torch
import pytest
from marqo.errors import BadRequestError


@pytest.mark.largemodel
@pytest.mark.skipif(torch.cuda.is_available() is False, reason="We skip the large model test if we don't have cuda support")
class TestModelAuthLoadedS3(MarqoTestCase):
    """loads an s3 model loaded index, for tests """

    model_abs_path = None
    fake_access_key_id = '12345'
    fake_secret_key = 'this-is-a-secret'
    index_name_1 = "test-model-auth-index-1"
    s3_object_key = 'path/to/your/secret_model.pt'
    s3_bucket = 'your-bucket-name'
    custom_model_name = 'my_model'
    device = 'cuda'

    @classmethod
    def setUpClass(cls) -> None:
        """Simulates downloading a model from a private and using it in an
        add docs call
        """
        super().setUpClass()

        cls.endpoint = cls.authorized_url
        cls.generic_header = {"Content-type": "application/json"}

        try:
            tensor_search.delete_index(config=cls.config, index_name=cls.index_name_1)
        except IndexNotFoundError as s:
            pass

        cls.model_abs_path = get_s3_model_absolute_cache_path(
            S3Location(
                Key=cls.s3_object_key,
                Bucket=cls.s3_bucket
            ))
        _delete_file(cls.model_abs_path)

        model_properties = {
            "name": "ViT-B/32",
            "dimensions": 512,
            "model_location": {
                "s3": {
                    "Bucket": cls.s3_bucket,
                    "Key": cls.s3_object_key,
                },
                "auth_required": True
            },
            "type": "open_clip",
        }
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=cls.config, index_name=cls.index_name_1, index_settings=s3_settings)

        public_model_url = "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt"

        # Create a mock Boto3 client
        mock_s3_client = mock.MagicMock()

        # Mock the generate_presigned_url method of the mock Boto3 client with a real OpenCLIP model, so that
        # the rest of the logic works.
        mock_s3_client.generate_presigned_url.return_value = public_model_url

        # file should not yet exist:
        assert not os.path.isfile(cls.model_abs_path)

        with unittest.mock.patch('boto3.client', return_value=mock_s3_client) as mock_boto3_client:
            # Call the function that uses the generate_presigned_url method
            res = tensor_search.add_documents(config=cls.config, add_docs_params=AddDocsParams(
                index_name=cls.index_name_1, auto_refresh=True, docs=[{'a': 'b'}],
                model_auth=ModelAuth(
                    s3=S3Auth(aws_access_key_id=cls.fake_access_key_id, aws_secret_access_key=cls.fake_secret_key))
            ))
            assert not res['errors']

        assert os.path.isfile(cls.model_abs_path)

        mock_s3_client.generate_presigned_url.assert_called_with(
            'get_object',
            Params={'Bucket': 'your-bucket-name', 'Key': cls.s3_object_key}
        )
        mock_boto3_client.assert_called_once_with(
            's3',
            aws_access_key_id=cls.fake_access_key_id,
            aws_secret_access_key=cls.fake_secret_key,
            aws_session_token=None
        )

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        _delete_file(cls.model_abs_path)

    def test_after_downloading_auth_doesnt_matter(self):
        """on this instance, at least"""
        res = tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name_1, auto_refresh=True, docs=[{'c': 'd'}], device=self.device
        ))
        assert not res['errors']

    def test_after_downloading_doesnt_redownload(self):
        """on this instance, at least"""
        tensor_search.eject_model(model_name=self.custom_model_name, device=self.device)
        mock_req = mock.MagicMock()
        with mock.patch('urllib.request.urlopen', mock_req):
            res = tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1, auto_refresh=True, docs=[{'c': 'd'}],
                device=self.device
            ))
            assert not res['errors']
            mock_req.assert_not_called()
