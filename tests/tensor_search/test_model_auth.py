"""todos: host a public HF-based CLIP (non-OpenCLIP) model so that we can use it for mocks and tests

multiprocessing should be tested manually -problem with mocking (deadlock esque)
"""
from marqo.s2_inference.random_utils import Random
from marqo.s2_inference.s2_inference import _convert_vectorized_output
from marqo.tensor_search import tensor_search
from marqo.tensor_search.models.add_docs_objects import AddDocsParams
from marqo.tensor_search.models.private_models import S3Auth, ModelAuth, HfAuth
from marqo.errors import InvalidArgError, IndexNotFoundError, BadRequestError
from tests.marqo_test import MarqoTestCase
from marqo.s2_inference.model_downloading.from_s3 import get_s3_model_absolute_cache_path
from marqo.tensor_search.models.external_apis.s3 import S3Location
from unittest import mock
import unittest
from marqo.s2_inference.s2_inference import clear_loaded_models
from transformers import AutoModel, AutoTokenizer
from marqo.s2_inference.processing.custom_clip_utils import download_pretrained_from_url
from marqo.s2_inference.hf_utils import extract_huggingface_archive
import os
from marqo.errors import BadRequestError, ModelNotInCacheError
from marqo.tensor_search.models.api_models import BulkSearchQuery, BulkSearchQueryEntity
from marqo.s2_inference.configs import ModelCache
import shutil
from marqo.tensor_search.models.external_apis.hf import HfModelLocation
from marqo.tensor_search.models.private_models import ModelLocation
from pydantic.error_wrappers import ValidationError

def fake_vectorise(*args, **_kwargs):
    random_model = Random(model_name='blah', embedding_dim=512)
    return _convert_vectorized_output(random_model.encode(_kwargs['content']))

def fake_vectorise_384(*args, **_kwargs):
    random_model = Random(model_name='blah', embedding_dim=384)
    return _convert_vectorized_output(random_model.encode(_kwargs['content']))

def _delete_file(file_path):
    try:
        os.remove(file_path)
    except FileNotFoundError:
        pass

def _delete_directory(directory_path):
    try:
        shutil.rmtree(directory_path)
    except FileNotFoundError:
        pass
    
    
def _get_base_index_settings():
    return {
        "index_defaults": {
            "treat_urls_and_pointers_as_images": True,
            "model": 'my_model',
            "normalize_embeddings": True,
            # notice model properties aren't here. Each test has to add it
        }
    }

class TestModelAuthLoadedS3(MarqoTestCase):
    """loads an s3 model loaded index, for tests that don't need to redownload
    the model each time """

    model_abs_path = None
    fake_access_key_id = '12345'
    fake_secret_key = 'this-is-a-secret'
    index_name_1 = "test-model-auth-index-1"
    s3_object_key = 'path/to/your/secret_model.pt'
    s3_bucket = 'your-bucket-name'
    custom_model_name = 'my_model'
    device='cpu'

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

        # now the file exists
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
        tensor_search.eject_model(model_name=cls.custom_model_name, device=cls.device)

    def test_after_downloading_auth_doesnt_matter(self):
        """on this instance, at least"""
        res = tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name_1, auto_refresh=True, docs=[{'c': 'd'}]
        ))
        assert not res['errors']

    def test_after_downloading_doesnt_redownload(self):
        """on this instance, at least"""
        tensor_search.eject_model(model_name=self.custom_model_name, device=self.device)
        mods = tensor_search.get_loaded_models()['models']
        assert not any([m['model_name'] == 'my_model' for m in mods])
        mock_req = mock.MagicMock()
        with mock.patch('urllib.request.urlopen', mock_req):
            res = tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1, auto_refresh=True, docs=[{'c': 'd'}]
            ))
            assert not res['errors']
            mock_req.assert_not_called()
        mods = tensor_search.get_loaded_models()['models']
        assert any([m['model_name'] == 'my_model' for m in mods])

    def test_after_downloading_search_doesnt_redownload(self):
        """on this instance, at least"""
        tensor_search.eject_model(model_name=self.custom_model_name, device=self.device)
        mods = tensor_search.get_loaded_models()['models']
        assert not any([m['model_name'] == 'my_model' for m in mods])
        mock_req = mock.MagicMock()
        with mock.patch('urllib.request.urlopen', mock_req):
            res = tensor_search.search(config=self.config,
                index_name=self.index_name_1, text='hi'
            )
            assert 'hits' in res
            mock_req.assert_not_called()

        mods = tensor_search.get_loaded_models()['models']
        assert any([m['model_name'] == 'my_model' for m in mods])

class TestModelAuthOpenCLIP(MarqoTestCase):

    device = 'cpu'

    def setUp(self) -> None:
        self.endpoint = self.authorized_url
        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "test-model-auth-index-1"
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass

    def tearDown(self) -> None:
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass

    def test_model_auth_hf(self):
        """
        Does not yet assert that a file is downloaded
        """
        hf_object = "some_model.pt"
        hf_repo_name = "MyRepo/test-private"
        hf_token = "hf_some_secret_key"

        model_properties = {
            "name": "ViT-B/32",
            "dimensions": 512,
            "model_location": {
                "hf": {
                    "repo_id": hf_repo_name,
                    "filename": hf_object,
                },
                "auth_required": True
            },
            "type": "open_clip",
        }
        hf_settings = _get_base_index_settings()
        hf_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=hf_settings)

        mock_hf_hub_download = mock.MagicMock()
        mock_hf_hub_download.return_value = 'cache/path/to/model.pt'

        mock_open_clip_creat_model = mock.MagicMock()

        with unittest.mock.patch('open_clip.create_model_and_transforms', mock_open_clip_creat_model):
            with unittest.mock.patch('marqo.s2_inference.model_downloading.from_hf.hf_hub_download', mock_hf_hub_download):
                try:
                    tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                        index_name=self.index_name_1, auto_refresh=True, docs=[{'a': 'b'}],
                        model_auth=ModelAuth(hf=HfAuth(token=hf_token))))
                except BadRequestError as e:
                    # bad request due to no models actually being loaded
                    print(e)
                    pass

        mock_hf_hub_download.assert_called_once_with(
            token=hf_token,
            repo_id=hf_repo_name,
            filename=hf_object,
            cache_dir = None,
        )

        # is the open clip model being loaded with the expected args?
        called_with_expected_args = any(
            call.kwargs.get("pretrained") == "cache/path/to/model.pt"
            and call.kwargs.get("model_name") == "ViT-B/32"
            for call in mock_open_clip_creat_model.call_args_list
        )
        assert len(mock_open_clip_creat_model.call_args_list) == 1
        assert called_with_expected_args, "Expected call not found"

    def test_model_auth_s3_search(self):
        """The other test load from add_docs, we have to make sure it works for
         search"""

        s3_object_key = 'path/to/your/secret_model.pt'
        s3_bucket = 'your-bucket-name'

        model_abs_path = get_s3_model_absolute_cache_path(
            S3Location(
                Key=s3_object_key,
                Bucket=s3_bucket
        ))
        _delete_file(model_abs_path)

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
        s3_settings = _get_base_index_settings()
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
            res = tensor_search.search(
                config=self.config, text='hello', index_name=self.index_name_1,
                model_auth=ModelAuth(s3=S3Auth(aws_access_key_id=fake_access_key_id, aws_secret_access_key=fake_secret_key))
            )

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
        _delete_file(model_abs_path)

    def test_model_auth_hf_search(self):
        """The other test focused on add_docs. This focuses on search
        Does not yet assert that a file is downloaded
        """
        hf_object = "some_model.pt"
        hf_repo_name = "MyRepo/test-private"
        hf_token = "hf_some_secret_key"

        model_properties = {
            "name": "ViT-B/32",
            "dimensions": 512,
            "model_location": {
                "hf": {
                    "repo_id": hf_repo_name,
                    "filename": hf_object,
                },
                "auth_required": True
            },
            "type": "open_clip",
        }
        hf_settings = _get_base_index_settings()
        hf_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=hf_settings)

        mock_hf_hub_download = mock.MagicMock()
        mock_hf_hub_download.return_value = 'cache/path/to/model.pt'

        mock_open_clip_creat_model = mock.MagicMock()

        with unittest.mock.patch('open_clip.create_model_and_transforms', mock_open_clip_creat_model):
            with unittest.mock.patch('marqo.s2_inference.model_downloading.from_hf.hf_hub_download', mock_hf_hub_download):
                try:
                    res = tensor_search.search(
                        config=self.config, text='hello', index_name=self.index_name_1,
                        model_auth=ModelAuth(hf=HfAuth(token=hf_token)))
                except BadRequestError:
                    # bad request due to no models actually being loaded
                    pass

        mock_hf_hub_download.assert_called_once_with(
            token=hf_token,
            repo_id=hf_repo_name,
            filename=hf_object,
            cache_dir = None,
        )

        # is the open clip model being loaded with the expected args?
        called_with_expected_args = any(
            call.kwargs.get("pretrained") == "cache/path/to/model.pt"
            and call.kwargs.get("model_name") == "ViT-B/32"
            for call in mock_open_clip_creat_model.call_args_list
        )
        assert len(mock_open_clip_creat_model.call_args_list) == 1
        assert called_with_expected_args, "Expected call not found"

    def test_model_auth_mismatch_param_s3_ix(self):
        """There isn't validation for the hf because users may download public models this way"""
        s3_object_key = 'path/to/your/secret_model.pt'
        s3_bucket = 'your-bucket-name'

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
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)

        public_model_url = "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt"
        hf_token = 'hf_secret_token'

        # Create a mock Boto3 client
        mock_s3_client = mock.MagicMock()

        # Mock the generate_presigned_url method of the mock Boto3 client with a real OpenCLIP model, so that
        # the rest of the logic works.
        mock_s3_client.generate_presigned_url.return_value = public_model_url

        with unittest.mock.patch('boto3.client', return_value=mock_s3_client):
            with self.assertRaises(BadRequestError) as cm:
                tensor_search.search(
                    config=self.config, text='hello', index_name=self.index_name_1,
                    model_auth=ModelAuth(hf=HfAuth(token=hf_token)))

                self.assertIn("s3 authorisation information is required", str(cm.exception))

    def test_model_loads_from_all_add_docs_derivatives(self):
        """Does it work from add_docs, add_docs orchestrator and add_documents_mp?
        """
        s3_object_key = 'path/to/your/secret_model.pt'
        s3_bucket = 'your-bucket-name'

        fake_access_key_id = '12345'
        fake_secret_key = 'this-is-a-secret'

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
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)

        for add_docs_method, kwargs in [
                (tensor_search.add_documents_orchestrator, {'batch_size': 10}),
            ]:
            try:
                tensor_search.eject_model(model_name='my_model' ,device=self.device)
            except ModelNotInCacheError:
                pass
            # Create a mock Boto3 client
            mock_s3_client = mock.MagicMock()

            # Mock the generate_presigned_url method of the mock Boto3 client with a real OpenCLIP model, so that
            # the rest of the logic works.
            mock_s3_client.generate_presigned_url.return_value = "https://some_non_existent_model.pt"

            with unittest.mock.patch('boto3.client', return_value=mock_s3_client) as mock_boto3_client:
                with self.assertRaises(BadRequestError) as cm:
                    with unittest.mock.patch(
                        'marqo.s2_inference.processing.custom_clip_utils.download_pretrained_from_url'
                    ) as mock_download_pretrained_from_url:
                        add_docs_method(
                            config=self.config,
                            add_docs_params=AddDocsParams(
                                index_name=self.index_name_1,
                                model_auth=ModelAuth(s3=S3Auth(
                                    aws_access_key_id=fake_access_key_id,
                                    aws_secret_access_key=fake_secret_key)),
                                auto_refresh=True,
                                docs=[{f'Title': "something {i} good"} for i in range(20)]
                            ),
                            **kwargs
                        )
            mock_download_pretrained_from_url.assert_called_once_with(
                url='https://some_non_existent_model.pt', cache_dir=None, cache_file_name='secret_model.pt')
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
            mock_download_pretrained_from_url.reset_mock()
            mock_s3_client.reset_mock()
            mock_boto3_client.reset_mock()

    def test_model_loads_from_multi_search(self):
        s3_object_key = 'path/to/your/secret_model.pt'
        s3_bucket = 'your-bucket-name'

        fake_access_key_id = '12345'
        fake_secret_key = 'this-is-a-secret'

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
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)

        random_model = Random(model_name='blah', embedding_dim=512)

        try:
            tensor_search.eject_model(model_name='my_model', device=self.device)
        except ModelNotInCacheError:
            pass
        # Create a mock Boto3 client
        mock_s3_client = mock.MagicMock()

        # Mock the generate_presigned_url method of the mock Boto3 client with a real OpenCLIP model, so that
        # the rest of the logic works.
        mock_s3_client.generate_presigned_url.return_value = "https://some_non_existent_model.pt"

        with unittest.mock.patch('marqo.s2_inference.s2_inference.vectorise',
                                 side_effect=fake_vectorise) as mock_vectorise:
            model_auth = ModelAuth(
                s3=S3Auth(
                    aws_access_key_id=fake_access_key_id,
                    aws_secret_access_key=fake_secret_key)
            )
            res = tensor_search.search(
                index_name=self.index_name_1,
                config=self.config,
                model_auth=model_auth,
                text={
                    (f"https://raw.githubusercontent.com/marqo-ai/"
                     f"marqo-api-tests/mainline/assets/ai_hippo_realistic.png"): 0.3,
                    'my text': -1.3
                },
            )
            assert 'hits' in res
            mock_vectorise.assert_called()
            assert len(mock_vectorise.call_args_list) > 0
            for _args, _kwargs in mock_vectorise.call_args_list:
                assert _kwargs['model_properties']['model_location'] == {
                    "s3": {
                        "Bucket": s3_bucket,
                        "Key": s3_object_key,
                    },
                    "auth_required": True
                }
                assert _kwargs['model_auth'] == model_auth

    def test_model_loads_from_multimodal_combination(self):
        s3_object_key = 'path/to/your/secret_model.pt'
        s3_bucket = 'your-bucket-name'

        fake_access_key_id = '12345'
        fake_secret_key = 'this-is-a-secret'

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
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)

        random_model = Random(model_name='blah', embedding_dim=512)


        for add_docs_method, kwargs in [
            (tensor_search.add_documents_orchestrator, {'batch_size': 10}),
            (tensor_search.add_documents, {})
        ]:
            try:
                tensor_search.eject_model(model_name='my_model', device=self.device)
            except ModelNotInCacheError:
                pass
            # Create a mock Boto3 client
            mock_s3_client = mock.MagicMock()

            # Mock the generate_presigned_url method of the mock Boto3 client with a real OpenCLIP model, so that
            # the rest of the logic works.
            mock_s3_client.generate_presigned_url.return_value = "https://some_non_existent_model.pt"

            with unittest.mock.patch('marqo.s2_inference.s2_inference.vectorise', side_effect=fake_vectorise) as mock_vectorise:
                model_auth = ModelAuth(
                    s3=S3Auth(
                    aws_access_key_id=fake_access_key_id,
                    aws_secret_access_key=fake_secret_key)
                )
                res = add_docs_method(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=self.index_name_1,
                        model_auth=model_auth,
                        auto_refresh=True,
                        docs=[{
                            'my_combination_field': {
                                'my_image': f"https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
                                'some_text': f"my text {i}"}} for i in range(20)],
                        mappings={
                            "my_combination_field": {
                                "type": "multimodal_combination",
                                "weights": {
                                    "my_image": 0.5,
                                    "some_text": 0.5
                                }
                            }
                        }
                    ),
                    **kwargs
                )
                if isinstance(res, list):
                    assert all([not batch_res ['errors'] for batch_res in res])
                else:
                    assert not res['errors']
                mock_vectorise.assert_called()
                for _args, _kwargs in mock_vectorise.call_args_list:
                    assert _kwargs['model_properties']['model_location'] == {
                        "s3": {
                            "Bucket": s3_bucket,
                            "Key": s3_object_key,
                        },
                        "auth_required": True
                    }
                    assert _kwargs['model_auth'] == model_auth

    def test_no_creds_error(self):
        """in s3, if there aren't creds"""
        s3_object_key = 'path/to/your/secret_model.pt'
        s3_bucket = 'your-bucket-name'

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
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)

        public_model_url = "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt"
        hf_token = 'hf_secret_token'

        # Create a mock Boto3 client
        mock_s3_client = mock.MagicMock()

        # Mock the generate_presigned_url method of the mock Boto3 client with a real OpenCLIP model, so that
        # the rest of the logic works.
        mock_s3_client.generate_presigned_url.return_value = public_model_url

        with unittest.mock.patch('boto3.client', return_value=mock_s3_client):
            with self.assertRaises(BadRequestError) as cm:
                tensor_search.search(
                    config=self.config, text='hello', index_name=self.index_name_1,
                )
                self.assertIn("s3 authorisation information is required", str(cm.exception))


        with unittest.mock.patch('boto3.client', return_value=mock_s3_client):
            with self.assertRaises(BadRequestError) as cm2:
                res = tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=self.index_name_1, auto_refresh=True,
                        docs=[{'title': 'blah blah'}]
                    )
                )
            self.assertIn("s3 authorisation information is required", str(cm2.exception))

    def test_bad_creds_error_s3(self):
        """in s3 if creds aren't valid. Ensure a helpful error"""
        s3_object_key = 'path/to/your/secret_model.pt'
        s3_bucket = 'your-bucket-name'

        fake_access_key_id = '12345'
        fake_secret_key = 'this-is-a-secret'

        model_auth = ModelAuth(
            s3=S3Auth(
                aws_access_key_id=fake_access_key_id,
                aws_secret_access_key=fake_secret_key)
        )

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
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)

        with self.assertRaises(BadRequestError) as cm:
            tensor_search.search(
                config=self.config, text='hello', index_name=self.index_name_1,
                model_auth=model_auth
            )
        self.assertIn("403 error when trying to retrieve model from s3", str(cm.exception))

        with self.assertRaises(BadRequestError) as cm2:
            res = tensor_search.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.index_name_1, auto_refresh=True,
                    docs=[{'title': 'blah blah'}], model_auth=model_auth
                )
            )
        self.assertIn("403 error when trying to retrieve model from s3", str(cm2.exception))

    def test_non_existent_hf_location(self):
        hf_object = "some_model.pt"
        hf_repo_name = "MyRepo/test-private"
        hf_token = "hf_some_secret_key"

        model_auth = ModelAuth(
            hf=HfAuth(token=hf_token)
        )

        model_properties = {
            "name": "ViT-B/32",
            "dimensions": 512,
            "model_location": {
                "hf": {
                    "repo_id": hf_repo_name,
                    "filename": hf_object,
                },
                "auth_required": True
            },
            "type": "open_clip",
        }
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)

        with self.assertRaises(BadRequestError) as cm:
            tensor_search.search(
                config=self.config, text='hello', index_name=self.index_name_1,
                model_auth=model_auth
            )

        self.assertIn("Could not find the specified Hugging Face model repository.", str(cm.exception))

        with self.assertRaises(BadRequestError) as cm2:
            res = tensor_search.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.index_name_1, auto_refresh=True,
                    docs=[{'title': 'blah blah'}], model_auth=model_auth
                )
            )
        self.assertIn("Could not find the specified Hugging Face model repository.", str(cm.exception))

    def test_bad_creds_error_hf(self):
        """the model and repo do exist, but creds are bad. raises the same type of error
        as the previous one. """
        hf_object = "dummy_model.pt"
        hf_repo_name = "Marqo/test-private"
        hf_token = "hf_some_secret_key"

        model_auth = ModelAuth(
            hf=HfAuth(token=hf_token)
        )

        model_properties = {
            "name": "ViT-B/32",
            "dimensions": 512,
            "model_location": {
                "hf": {
                    "repo_id": hf_repo_name,
                    "filename": hf_object,
                },
                "auth_required": True
            },
            "type": "open_clip",
        }
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)

        with self.assertRaises(BadRequestError) as cm:
            tensor_search.search(
                config=self.config, text='hello', index_name=self.index_name_1,
                model_auth=model_auth
            )
        self.assertIn("Could not find the specified Hugging Face model repository.", str(cm.exception))

        with self.assertRaises(BadRequestError) as cm2:
            res = tensor_search.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.index_name_1, auto_refresh=True,
                    docs=[{'title': 'blah blah'}], model_auth=model_auth
                )
            )
        self.assertIn("Could not find the specified Hugging Face model repository.", str(cm.exception))

    def test_bulk_search(self):
        """Does it work with bulk search, including multi search
        """
        s3_object_key = 'path/to/your/secret_model.pt'
        s3_bucket = 'your-bucket-name'

        fake_access_key_id = '12345'
        fake_secret_key = 'this-is-a-secret'

        model_auth = ModelAuth(
            s3=S3Auth(
                aws_access_key_id=fake_access_key_id,
                aws_secret_access_key=fake_secret_key)
        )

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
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)

        for bulk_search_query in [
                BulkSearchQuery(queries=[
                    BulkSearchQueryEntity(
                        index=self.index_name_1, q="match", searchMethod="TENSOR",
                        modelAuth=model_auth
                    ),
                    BulkSearchQueryEntity(
                        index=self.index_name_1, q={"random text": 0.5, "other_text": -0.3},
                        searchableAttributes=["abc"], searchMethod="TENSOR",
                        modelAuth=model_auth
                    ),
                ]),
                BulkSearchQuery(queries=[
                    BulkSearchQueryEntity(
                        index=self.index_name_1, q={"random text": 0.5, "other_text": -0.3},
                        searchableAttributes=["abc"], searchMethod="TENSOR",
                        modelAuth=model_auth
                    ),
                ])
            ]:
            try:
                tensor_search.eject_model(model_name='my_model' ,device=self.device)
            except ModelNotInCacheError:
                pass
            # Create a mock Boto3 client
            mock_s3_client = mock.MagicMock()

            # Mock the generate_presigned_url method of the mock Boto3 client with a real OpenCLIP model, so that
            # the rest of the logic works.
            mock_s3_client.generate_presigned_url.return_value = "https://some_non_existent_model.pt"

            with unittest.mock.patch('boto3.client', return_value=mock_s3_client) as mock_boto3_client:
                with self.assertRaises(InvalidArgError) as cm:
                    with unittest.mock.patch(
                        'marqo.s2_inference.processing.custom_clip_utils.download_pretrained_from_url'
                    ) as mock_download_pretrained_from_url:
                        tensor_search.bulk_search(
                            query=bulk_search_query,
                            marqo_config=self.config,
                        )
            mock_download_pretrained_from_url.assert_called_once_with(
                url='https://some_non_existent_model.pt', cache_dir=None, cache_file_name='secret_model.pt')
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

            mock_download_pretrained_from_url.reset_mock()
            mock_s3_client.reset_mock()
            mock_boto3_client.reset_mock()

    def test_bulk_search_vectorise(self):
        """are the calls to vectorise expected? work with bulk search, including multi search
        """
        s3_object_key = 'path/to/your/secret_model.pt'
        s3_bucket = 'your-bucket-name'

        fake_access_key_id = '12345'
        fake_secret_key = 'this-is-a-secret'

        model_auth = ModelAuth(
            s3=S3Auth(
                aws_access_key_id=fake_access_key_id,
                aws_secret_access_key=fake_secret_key)
        )

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
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)

        for bulk_search_query in [
                BulkSearchQuery(queries=[
                    BulkSearchQueryEntity(
                        index=self.index_name_1, q="match", searchMethod="TENSOR",
                        modelAuth=model_auth
                    ),
                    BulkSearchQueryEntity(
                        index=self.index_name_1, q={"random text": 0.5, "other_text": -0.3},
                        searchableAttributes=["abc"], searchMethod="TENSOR",
                        modelAuth=model_auth
                    ),
                ]),
                BulkSearchQuery(queries=[
                    BulkSearchQueryEntity(
                        index=self.index_name_1, q={"random text": 0.5, "other_text": -0.3},
                        searchableAttributes=["abc"], searchMethod="TENSOR",
                        modelAuth=model_auth
                    ),
                ])
            ]:
            try:
                tensor_search.eject_model(model_name='my_model' ,device=self.device)
            except ModelNotInCacheError:
                pass
            # Create a mock Boto3 client
            mock_s3_client = mock.MagicMock()

            # Mock the generate_presigned_url method of the mock Boto3 client with a real OpenCLIP model, so that
            # the rest of the logic works.
            mock_s3_client.generate_presigned_url.return_value = "https://some_non_existent_model.pt"

            with unittest.mock.patch('marqo.s2_inference.s2_inference.vectorise',
                                     side_effect=fake_vectorise) as mock_vectorise:
                        tensor_search.bulk_search(
                            query=bulk_search_query,
                            marqo_config=self.config,
                        )
            mock_vectorise.assert_called()
            for _args, _kwargs in mock_vectorise.call_args_list:
                assert _kwargs['model_properties']['model_location'] == {
                    "s3": {
                        "Bucket": s3_bucket,
                        "Key": s3_object_key,
                    },
                    "auth_required": True
                }
                assert _kwargs['model_auth'] == model_auth

            mock_vectorise.reset_mock()

    def test_lexical_with_auth(self):
        """should just skip"""

    def test_public_s3_no_auth(self):
        """
        TODO
        """

    def test_public_hf_no_auth(self):
        """
        TODO
        """

    def test_open_clip_reg_clip(self):
        """both normal and open clip
        TODO: normal CLIP
        """



class TestModelAuthDownloadAndExtractS3HFModel(MarqoTestCase):
    """Tests for the downloading and archive extracting process for s3 and hf models"""

    model_abs_path = None
    fake_access_key_id = '12345'
    fake_secret_key = 'this-is-a-secret'
    index_name_1 = "test-model-auth-index-1"
    s3_object_key = 'path/to/your/secret_model.zip'
    s3_bucket = 'your-bucket-name'
    custom_model_name = 'my_model'
    device = 'cpu'

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
            ), download_dir=ModelCache.hf_cache_path)

        _delete_file(cls.model_abs_path)

        model_properties = {
            "dimensions": 384,
            "model_location": {
                "s3": {
                    "Bucket": cls.s3_bucket,
                    "Key": cls.s3_object_key,
                },
                "auth_required": True
            },
            "type": "hf",
        }
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=cls.config, index_name=cls.index_name_1, index_settings=s3_settings)

        public_model_url = "https://marqo-cache-sentence-transformers.s3.us-west-2.amazonaws.com/all-MiniLM-L6-v1/all-MiniLM-L6-v1.zip"

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

        # now the file exists
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
        tensor_search.eject_model(model_name=cls.custom_model_name, device=cls.device)

    def test_after_downloading_auth_doesnt_matter(self):
        """on this instance, at least"""
        res = tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
            index_name=self.index_name_1, auto_refresh=True, docs=[{'c': 'd'}]
        ))
        assert not res['errors']

    def test_after_downloading_doesnt_redownload(self):
        """on this instance, at least"""
        tensor_search.eject_model(model_name=self.custom_model_name, device=self.device)
        mods = tensor_search.get_loaded_models()['models']
        assert not any([m['model_name'] == 'my_model' for m in mods])
        mock_req = mock.MagicMock()
        with mock.patch('urllib.request.urlopen', mock_req):
            res = tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1, auto_refresh=True, docs=[{'c': 'd'}]
            ))
            assert not res['errors']
            mock_req.assert_not_called()
        mods = tensor_search.get_loaded_models()['models']
        assert any([m['model_name'] == 'my_model' for m in mods])

    def test_after_downloading_search_doesnt_redownload(self):
        """on this instance, at least"""
        tensor_search.eject_model(model_name=self.custom_model_name, device=self.device)
        mods = tensor_search.get_loaded_models()['models']
        assert not any([m['model_name'] == 'my_model' for m in mods])
        mock_req = mock.MagicMock()
        with mock.patch('urllib.request.urlopen', mock_req):
            res = tensor_search.search(config=self.config,
                                       index_name=self.index_name_1, text='hi'
                                       )
            assert 'hits' in res
            mock_req.assert_not_called()

        mods = tensor_search.get_loaded_models()['models']
        assert any([m['model_name'] == 'my_model' for m in mods])


class TestModelAuthlLoadForHFModelBasic(MarqoTestCase):
    """
    This class tests the following scenarios:

    1. Load from huggingface zip file, with auth
    2. Load from huggingface zip file, without auth

    3. Load from s3 zip file, with auth
    4. Load from public url, without auth

    5. Load from huggingface repo, with auth
    61. Load from huggingface, without auth, using hf.repo_id
    62. Load from huggingface, without auth, using name

    for both search and add_document calls
    """

    device = 'cpu'

    def setUp(self) -> None:
        self.endpoint = self.authorized_url
        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "test-model-auth-index-1"
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass


    def tearDown(self) -> None:
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass

        clear_loaded_models()

    def test_1_load_model_from_hf_zip_file_with_auth_search(self):
        """
        Does not yet assert that a file is downloaded
        """
        hf_object = "some_model_archive.zip"
        hf_repo_name = "MyRepo/test-private"
        hf_token = "hf_some_secret_key"

        model_properties = {
            #"name": "bulabulabula", Note: name is not a required field for HF models.
            "dimensions": 384,
            "model_location": {
                "hf": {
                    "repo_id": hf_repo_name,
                    "filename": hf_object,
                },
                "auth_required": True
            },
            "type": "hf",
        }
        hf_settings = _get_base_index_settings()
        hf_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=hf_settings)

        mock_hf_hub_download = mock.MagicMock()
        mock_hf_hub_download.return_value = 'cache/path/to/model.zip'

        mock_extract_huggingface_archive = mock.MagicMock()
        mock_extract_huggingface_archive.return_value = 'cache/path/to/model/'

        mock_automodel_from_pretrained = mock.MagicMock()
        mock_autotokenizer_from_pretrained = mock.MagicMock()

        with unittest.mock.patch('transformers.AutoModel.from_pretrained', mock_automodel_from_pretrained):
            with unittest.mock.patch('transformers.AutoTokenizer.from_pretrained', mock_autotokenizer_from_pretrained):
                with unittest.mock.patch('marqo.s2_inference.model_downloading.from_hf.hf_hub_download', mock_hf_hub_download):
                    with unittest.mock.patch("marqo.s2_inference.hf_utils.extract_huggingface_archive", mock_extract_huggingface_archive):
                        try:
                            res = tensor_search.search(
                                config=self.config, text='hello', index_name=self.index_name_1,
                                model_auth=ModelAuth(hf=HfAuth(token=hf_token))
                            )
                        except KeyError as e:
                            # KeyError as this is not a real model. It does not have an attention_mask
                            assert "attention_mask" in str(e)
                            pass

        mock_hf_hub_download.assert_called_once_with(
            token=hf_token,
            repo_id=hf_repo_name,
            filename=hf_object,
            cache_dir = ModelCache.hf_cache_path,
        )

        # is the hf model being loaded with the expected args?
        assert len(mock_automodel_from_pretrained.call_args_list) == 1
        assert mock_automodel_from_pretrained.call_args_list[0][0][0] == 'cache/path/to/model/', "Expected call not found"

        # is the zip file being extracted with the expected args?
        assert len(mock_extract_huggingface_archive.call_args_list) == 1
        assert mock_extract_huggingface_archive.call_args_list[0][0][0] == 'cache/path/to/model.zip', "Expected call not found"

    def test_2_load_model_from_hf_zip_file_without_auth_search(self):
        """
        Does not yet assert that a file is downloaded
        """
        hf_object = "some_model.zip"
        hf_repo_name = "MyRepo/test-private"

        model_properties = {
            "dimensions": 384,
            "model_location": {
                "hf": {
                    "repo_id": hf_repo_name,
                    "filename": hf_object,
                },
                "auth_required": False
            },
            "type": "hf",
        }
        hf_settings = _get_base_index_settings()
        hf_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=hf_settings)

        mock_hf_hub_download = mock.MagicMock()
        mock_hf_hub_download.return_value = 'cache/path/to/model.zip'

        mock_extract_huggingface_archive = mock.MagicMock()
        mock_extract_huggingface_archive.return_value = 'cache/path/to/model/'

        mock_automodel_from_pretrained = mock.MagicMock()
        mock_autotokenizer_from_pretrained = mock.MagicMock()

        with unittest.mock.patch('transformers.AutoModel.from_pretrained', mock_automodel_from_pretrained):
            with unittest.mock.patch('transformers.AutoTokenizer.from_pretrained', mock_autotokenizer_from_pretrained):
                with unittest.mock.patch('marqo.s2_inference.model_downloading.from_hf.hf_hub_download', mock_hf_hub_download):
                    with unittest.mock.patch("marqo.s2_inference.hf_utils.extract_huggingface_archive", mock_extract_huggingface_archive):
                        try:
                            res = tensor_search.search(
                                config=self.config, text='hello', index_name=self.index_name_1,)
                        except KeyError as e:
                            # KeyError as this is not a real model. It does not have an attention_mask
                            assert "attention_mask" in str(e)
                            pass

        mock_hf_hub_download.assert_called_once_with(
            repo_id=hf_repo_name,
            filename=hf_object,
            cache_dir = ModelCache.hf_cache_path,
        )

        # is the hf model being loaded with the expected args?
        assert len(mock_automodel_from_pretrained.call_args_list) == 1
        assert mock_automodel_from_pretrained.call_args_list[0][0][0] == 'cache/path/to/model/', "Expected call not found"

        # is the hf tokenizer being loaded with the expected args?
        assert len(mock_autotokenizer_from_pretrained.call_args_list) == 1
        assert mock_autotokenizer_from_pretrained.call_args_list[0][0][0] == 'cache/path/to/model/', "Expected call not found"

        # is the zip file being extracted with the expected args?
        assert len(mock_extract_huggingface_archive.call_args_list) == 1
        assert mock_extract_huggingface_archive.call_args_list[0][0][0] == 'cache/path/to/model.zip', "Expected call not found"

    def test_3_load_model_from_s3_zip_file_with_auth_search(self):
        s3_object_key = 'path/to/your/secret_model.pt'
        s3_bucket = 'your-bucket-name'

        _delete_file(os.path.join(ModelCache.hf_cache_path, os.path.basename(s3_object_key)))

        model_properties = {
            "dimensions": 384,
            "model_location": {
                "s3": {
                    "Bucket": s3_bucket,
                    "Key": s3_object_key,
                },
                "auth_required": True
            },
            "type": "hf",
        }
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)

        fake_access_key_id = '12345'
        fake_secret_key = 'this-is-a-secret'
        public_model_url = "https://dummy/url/for/model.zip"

        # Create a mock Boto3 client
        mock_s3_client = mock.MagicMock()

        # Mock the generate_presigned_url method of the mock Boto3 client to return a dummy URL
        mock_s3_client.generate_presigned_url.return_value = public_model_url

        mock_download_pretrained_from_url = mock.MagicMock()
        mock_download_pretrained_from_url.return_value = 'cache/path/to/model.zip'

        mock_extract_huggingface_archive = mock.MagicMock()
        mock_extract_huggingface_archive.return_value = 'cache/path/to/model/'

        mock_automodel_from_pretrained = mock.MagicMock()
        mock_autotokenizer_from_pretrained = mock.MagicMock()

        with unittest.mock.patch('transformers.AutoModel.from_pretrained', mock_automodel_from_pretrained):
            with unittest.mock.patch('transformers.AutoTokenizer.from_pretrained',mock_autotokenizer_from_pretrained):
                with unittest.mock.patch('boto3.client', return_value=mock_s3_client) as mock_boto3_client:
                    with unittest.mock.patch("marqo.s2_inference.processing.custom_clip_utils.download_pretrained_from_url", mock_download_pretrained_from_url):
                        with unittest.mock.patch("marqo.s2_inference.hf_utils.extract_huggingface_archive", mock_extract_huggingface_archive):
                            try:
                                res = tensor_search.search(
                                    config=self.config, text='hello', index_name=self.index_name_1,
                                    model_auth=ModelAuth(s3=S3Auth(aws_access_key_id=fake_access_key_id, aws_secret_access_key=fake_secret_key))
                                )
                            except KeyError as e:
                                # KeyError as this is not a real model. It does not have an attention_mask
                                assert "attention_mask" in str(e)
                                pass

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
        mock_autotokenizer_from_pretrained.assert_called_once_with(
            "cache/path/to/model/",
        )

        mock_download_pretrained_from_url.assert_called_once_with(
            url = public_model_url,
            cache_dir = ModelCache.hf_cache_path,
            cache_file_name = os.path.basename(s3_object_key)
        )

        mock_extract_huggingface_archive.assert_called_once_with(
            "cache/path/to/model.zip",
        )

    def test_4_load_model_from_public_url_zip_file_search(self):
        public_url = "https://marqo-cache-sentence-transformers.s3.us-west-2.amazonaws.com/all-MiniLM-L6-v1/all-MiniLM-L6-v1.zip"

        model_properties = {
            "dimensions": 384,
            "url" : public_url,
            "type": "hf",
        }
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1,
                                          index_settings=s3_settings)

        mock_extract_huggingface_archive = mock.MagicMock(side_effect=extract_huggingface_archive)
        mock_automodel_from_pretrained = mock.MagicMock(side_effect=AutoModel.from_pretrained)
        mock_download = mock.MagicMock(side_effect=download_pretrained_from_url)

        with mock.patch('transformers.AutoModel.from_pretrained', new=mock_automodel_from_pretrained):
            with mock.patch('marqo.s2_inference.processing.custom_clip_utils.download_pretrained_from_url', new=mock_download):
                with mock.patch("marqo.s2_inference.hf_utils.extract_huggingface_archive", new=mock_extract_huggingface_archive):
                    res = tensor_search.search(config=self.config, text='hello', index_name=self.index_name_1)

        assert len(mock_extract_huggingface_archive.call_args_list) == 1
        assert mock_extract_huggingface_archive.call_args_list[0][0][0] == (ModelCache.hf_cache_path + os.path.basename(public_url))

        assert len(mock_download.call_args_list) == 1
        _, call_kwargs = mock_download.call_args_list[0]
        assert call_kwargs['url'] == public_url

        assert len(mock_automodel_from_pretrained.call_args_list) == 1
        assert mock_automodel_from_pretrained.call_args_list[0][0][0] == ModelCache.hf_cache_path + os.path.basename(public_url).replace('.zip', '')

    def test_5_load_model_from_private_hf_repo_with_auth_search(self):
        private_repo_name = "your-private-repo"
        hf_token = "some-secret-token"

        model_properties = {
            "dimensions": 384,
            "model_location": {
                "auth_required": True,
                "hf": {"repo_id": private_repo_name}
            },
            "type": "hf",
        }

        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1,
                                          index_settings=s3_settings)

        # Redirect the private model to a public one to finish the test
        mock_automodel_from_pretrained = mock.MagicMock()
        mock_automodel_from_pretrained.return_value = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v1")

        mock_autotokenizer_from_pretrained = mock.MagicMock()
        mock_autotokenizer_from_pretrained.return_value = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v1")

        with unittest.mock.patch("transformers.AutoModel.from_pretrained",  mock_automodel_from_pretrained):
            with unittest.mock.patch("transformers.AutoTokenizer.from_pretrained", mock_autotokenizer_from_pretrained):
                res = tensor_search.search(config=self.config, text='hello', index_name=self.index_name_1,
                                           model_auth=ModelAuth(hf=HfAuth(token=hf_token)))

        mock_automodel_from_pretrained.assert_called_once_with(
            private_repo_name, use_auth_token=hf_token, cache_dir=ModelCache.hf_cache_path
        )

        mock_autotokenizer_from_pretrained.assert_called_once_with(
            private_repo_name, use_auth_token=hf_token
        )

    def test_61_load_model_from_public_hf_repo_without_auth_using_repo_id_search(self):
        public_repo_name = "sentence-transformers/all-MiniLM-L6-v1"
        hf_token = "some-secret-token"

        model_properties = {
            "dimensions": 384,
            "model_location": {
                "auth_required": False,
                "hf": {"repo_id": public_repo_name}
            },
            "type": "hf",
        }

        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1,
                                          index_settings=s3_settings)

        # Redirect the private model to a public one to finish the test
        mock_automodel_from_pretrained = mock.MagicMock()
        mock_automodel_from_pretrained.return_value = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v1")

        mock_autotokenizer_from_pretrained = mock.MagicMock()
        mock_autotokenizer_from_pretrained.return_value = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v1")

        with unittest.mock.patch("transformers.AutoModel.from_pretrained",  mock_automodel_from_pretrained):
            with unittest.mock.patch("transformers.AutoTokenizer.from_pretrained", mock_autotokenizer_from_pretrained):
                res = tensor_search.search(config=self.config, text='hello', index_name=self.index_name_1)

        mock_automodel_from_pretrained.assert_called_once_with(
            public_repo_name, use_auth_token=None, cache_dir=ModelCache.hf_cache_path
        )

        mock_autotokenizer_from_pretrained.assert_called_once_with(
            public_repo_name, use_auth_token=None
        )

    def test_62_load_model_from_public_hf_repo_without_auth_using_name_search(self):
        public_repo_name = "sentence-transformers/all-MiniLM-L6-v1"

        model_properties = {
            "name": public_repo_name,
            "dimensions": 384,
            "type": "hf",
        }
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1,
                                          index_settings=s3_settings)

        mock_automodel_from_pretrained = mock.MagicMock(side_effect=AutoModel.from_pretrained)

        with mock.patch('transformers.AutoModel.from_pretrained', new=mock_automodel_from_pretrained):
                    res = tensor_search.search(config=self.config, text='hello', index_name=self.index_name_1)

        assert mock_automodel_from_pretrained.assert_called_once_with(
            public_repo_name, use_auth_token=None, cache_dir=ModelCache.hf_cache_path
        )

    def test_1_load_model_from_hf_zip_file_with_auth_add_documents(self):
        """
        Does not yet assert that a file is downloaded
        """
        hf_object = "some_model.pt"
        hf_repo_name = "MyRepo/test-private"
        hf_token = "hf_some_secret_key"

        model_properties = {
            #"name": "ViT-B/32", Note, name is not a required field for HF models.
            "dimensions": 384,
            "model_location": {
                "hf": {
                    "repo_id": hf_repo_name,
                    "filename": hf_object,
                },
                "auth_required": True
            },
            "type": "hf",
        }
        hf_settings = _get_base_index_settings()
        hf_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=hf_settings)

        mock_hf_hub_download = mock.MagicMock()
        mock_hf_hub_download.return_value = 'cache/path/to/model.zip'

        mock_extract_huggingface_archive = mock.MagicMock()
        mock_extract_huggingface_archive.return_value = 'cache/path/to/model/'

        mock_automodel_from_pretrained = mock.MagicMock()
        mock_autotokenizer_from_pretrained = mock.MagicMock()

        with unittest.mock.patch('transformers.AutoModel.from_pretrained', mock_automodel_from_pretrained):
            with unittest.mock.patch('transformers.AutoTokenizer.from_pretrained', mock_autotokenizer_from_pretrained):
                with unittest.mock.patch('marqo.s2_inference.model_downloading.from_hf.hf_hub_download', mock_hf_hub_download):
                    with unittest.mock.patch("marqo.s2_inference.hf_utils.extract_huggingface_archive", mock_extract_huggingface_archive):
                        try:
                            tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                                index_name=self.index_name_1, auto_refresh=True, docs=[{'a': 'b'}],
                                model_auth=ModelAuth(hf=HfAuth(token=hf_token))))
                        except KeyError as e:
                            # KeyError as this is not a real model. It does not have an attention_mask
                            assert "attention_mask" in str(e)
                            pass

        mock_hf_hub_download.assert_called_once_with(
            token=hf_token,
            repo_id=hf_repo_name,
            filename=hf_object,
            cache_dir = ModelCache.hf_cache_path
        )

        # is the hf model being loaded with the expected args?
        assert len(mock_automodel_from_pretrained.call_args_list) == 1
        assert mock_automodel_from_pretrained.call_args_list[0][0][0] == 'cache/path/to/model/', "Expected call not found"

        # is the zip file being extracted with the expected args?
        assert len(mock_extract_huggingface_archive.call_args_list) == 1
        assert mock_extract_huggingface_archive.call_args_list[0][0][0] == 'cache/path/to/model.zip', "Expected call not found"

    def test_2_load_model_from_hf_zip_file_without_auth_add_documents(self):
        """
        Does not yet assert that a file is downloaded
        """
        hf_object = "some_model.pt"
        hf_repo_name = "MyRepo/test-private"
        hf_token = "hf_some_secret_key"

        model_properties = {
            "dimensions": 384,
            "model_location": {
                "hf": {
                    "repo_id": hf_repo_name,
                    "filename": hf_object,
                },
                "auth_required": False
            },
            "type": "hf",
        }
        hf_settings = _get_base_index_settings()
        hf_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=hf_settings)

        mock_hf_hub_download = mock.MagicMock()
        mock_hf_hub_download.return_value = 'cache/path/to/model.zip'

        mock_extract_huggingface_archive = mock.MagicMock()
        mock_extract_huggingface_archive.return_value = 'cache/path/to/model/'

        mock_automodel_from_pretrained = mock.MagicMock()
        mock_autotokenizer_from_pretrained = mock.MagicMock()

        with unittest.mock.patch('transformers.AutoModel.from_pretrained', mock_automodel_from_pretrained):
            with unittest.mock.patch('transformers.AutoTokenizer.from_pretrained', mock_autotokenizer_from_pretrained):
                with unittest.mock.patch('marqo.s2_inference.model_downloading.from_hf.hf_hub_download', mock_hf_hub_download):
                    with unittest.mock.patch("marqo.s2_inference.hf_utils.extract_huggingface_archive", mock_extract_huggingface_archive):
                        try:
                            tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                                index_name=self.index_name_1, auto_refresh=True, docs=[{'a': 'b'}]))
                        except KeyError as e:
                            # KeyError as this is not a real model. It does not have an attention_mask
                            assert "attention_mask" in str(e)
                            pass

        mock_hf_hub_download.assert_called_once_with(
            repo_id=hf_repo_name,
            filename=hf_object,
            cache_dir = ModelCache.hf_cache_path
        )

        # is the hf model being loaded with the expected args?
        assert len(mock_automodel_from_pretrained.call_args_list) == 1
        assert mock_automodel_from_pretrained.call_args_list[0][0][0] == 'cache/path/to/model/', "Expected call not found"

        # is the zip file being extracted with the expected args?
        assert len(mock_extract_huggingface_archive.call_args_list) == 1
        assert mock_extract_huggingface_archive.call_args_list[0][0][0] == 'cache/path/to/model.zip', "Expected call not found"

    def test_3_load_model_from_s3_zip_file_with_auth_add_documents(self):
        def test_3_load_model_from_s3_zip_file_with_auth_search(self):
            s3_object_key = 'path/to/your/secret_model.pt'
            s3_bucket = 'your-bucket-name'

            _delete_file(os.path.join(ModelCache.hf_cache_path, os.path.basename(s3_object_key)))

            model_properties = {
                "dimensions": 384,
                "model_location": {
                    "s3": {
                        "Bucket": s3_bucket,
                        "Key": s3_object_key,
                    },
                    "auth_required": True
                },
                "type": "hf",
            }
            s3_settings = _get_base_index_settings()
            s3_settings['index_defaults']['model_properties'] = model_properties
            tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1,
                                              index_settings=s3_settings)

            fake_access_key_id = '12345'
            fake_secret_key = 'this-is-a-secret'
            public_model_url = "https://dummy/url/for/model.zip"

            # Create a mock Boto3 client
            mock_s3_client = mock.MagicMock()

            # Mock the generate_presigned_url method of the mock Boto3 client to return a dummy URL
            mock_s3_client.generate_presigned_url.return_value = public_model_url

            mock_download_pretrained_from_url = mock.MagicMock()
            mock_download_pretrained_from_url.return_value = 'cache/path/to/model.zip'

            mock_extract_huggingface_archive = mock.MagicMock()
            mock_extract_huggingface_archive.return_value = 'cache/path/to/model/'

            mock_automodel_from_pretrained = mock.MagicMock()
            mock_autotokenizer_from_pretrained = mock.MagicMock()

            with unittest.mock.patch('transformers.AutoModel.from_pretrained', mock_automodel_from_pretrained):
                with unittest.mock.patch('transformers.AutoTokenizer.from_pretrained',
                                         mock_autotokenizer_from_pretrained):
                    with unittest.mock.patch('boto3.client', return_value=mock_s3_client) as mock_boto3_client:
                        with unittest.mock.patch(
                                "marqo.s2_inference.processing.custom_clip_utils.download_pretrained_from_url",
                                mock_download_pretrained_from_url):
                            with unittest.mock.patch("marqo.s2_inference.hf_utils.extract_huggingface_archive",
                                                     mock_extract_huggingface_archive):
                                try:
                                    tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                                        index_name=self.index_name_1, auto_refresh=True, docs=[{'a': 'b'}],
                                        model_auth=ModelAuth(s3=S3Auth(aws_access_key_id=fake_access_key_id,
                                                                       aws_secret_access_key=fake_secret_key))))
                                except KeyError as e:
                                    # KeyError as this is not a real model. It does not have an attention_mask
                                    assert "attention_mask" in str(e)
                                    pass

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
            mock_autotokenizer_from_pretrained.assert_called_once_with(
                "cache/path/to/model/",
            )

            mock_download_pretrained_from_url.assert_called_once_with(
                url=public_model_url,
                cache_dir=ModelCache.hf_cache_path,
                cache_file_name=os.path.basename(s3_object_key)
            )

            mock_extract_huggingface_archive.assert_called_once_with(
                "cache/path/to/model.zip",
            )

    def test_4_load_model_from_public_url_zip_file_add_documents(self):
        public_url = "https://marqo-cache-sentence-transformers.s3.us-west-2.amazonaws.com/all-MiniLM-L6-v1/all-MiniLM-L6-v1.zip"

        model_properties = {
            # "name": "ViT-B/32", Note, name is not a required field for HF models.
            "dimensions": 384,
            "url" : public_url,
            "type": "hf",
        }
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1,
                                          index_settings=s3_settings)

        mock_extract_huggingface_archive = mock.MagicMock(side_effect=extract_huggingface_archive)
        mock_automodel_from_pretrained = mock.MagicMock(side_effect=AutoModel.from_pretrained)
        mock_download = mock.MagicMock(side_effect=download_pretrained_from_url)

        with mock.patch('transformers.AutoModel.from_pretrained', new=mock_automodel_from_pretrained):
            with mock.patch('marqo.s2_inference.processing.custom_clip_utils.download_pretrained_from_url', new=mock_download):
                with mock.patch("marqo.s2_inference.hf_utils.extract_huggingface_archive", new=mock_extract_huggingface_archive):
                    tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                        index_name=self.index_name_1, auto_refresh=True, docs=[{'a': 'b'}]))

        assert len(mock_extract_huggingface_archive.call_args_list) == 1
        assert mock_extract_huggingface_archive.call_args_list[0][0][0] == (ModelCache.hf_cache_path + os.path.basename(public_url))

        assert len(mock_download.call_args_list) == 1
        _, call_kwargs = mock_download.call_args_list[0]
        assert call_kwargs['url'] == public_url

        assert len(mock_automodel_from_pretrained.call_args_list) == 1
        assert mock_automodel_from_pretrained.call_args_list[0][0][0] == ModelCache.hf_cache_path + os.path.basename(public_url).replace('.zip', '')

    def test_5_load_model_from_private_hf_repo_with_auth_add_documents(self):
        private_repo_name = "your-private-repo"
        hf_token = "some-secret-token"

        model_properties = {
            "dimensions": 384,
            "model_location": {
                "auth_required": True,
                "hf": {"repo_id": private_repo_name}
            },
            "type": "hf",
        }

        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1,
                                          index_settings=s3_settings)

        # Redirect the private model to a public one to finish the test
        mock_automodel_from_pretrained = mock.MagicMock()
        mock_automodel_from_pretrained.return_value = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v1")

        mock_autotokenizer_from_pretrained = mock.MagicMock()
        mock_autotokenizer_from_pretrained.return_value = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v1")

        with unittest.mock.patch("transformers.AutoModel.from_pretrained",  mock_automodel_from_pretrained):
            with unittest.mock.patch("transformers.AutoTokenizer.from_pretrained", mock_autotokenizer_from_pretrained):
                tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.index_name_1, auto_refresh=True, docs=[{'a': 'b'}], model_auth=ModelAuth(hf=HfAuth(token=hf_token))))


        mock_automodel_from_pretrained.assert_called_once_with(
            private_repo_name, use_auth_token=hf_token, cache_dir=ModelCache.hf_cache_path
        )

        mock_autotokenizer_from_pretrained.assert_called_once_with(
            private_repo_name, use_auth_token=hf_token
        )


    def test_61_load_model_from_public_hf_repo_without_auth_using_repo_id_add_documents(self):
        public_repo_name = "sentence-transformers/all-MiniLM-L6-v1"
        hf_token = "some-secret-token"

        model_properties = {
            "dimensions": 384,
            "model_location": {
                "auth_required": False,
                "hf": {"repo_id": public_repo_name}
            },
            "type": "hf",
        }

        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1,
                                          index_settings=s3_settings)

        # Redirect the private model to a public one to finish the test
        mock_automodel_from_pretrained = mock.MagicMock()
        mock_automodel_from_pretrained.return_value = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v1")

        mock_autotokenizer_from_pretrained = mock.MagicMock()
        mock_autotokenizer_from_pretrained.return_value = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v1")

        with unittest.mock.patch("transformers.AutoModel.from_pretrained", mock_automodel_from_pretrained):
            with unittest.mock.patch("transformers.AutoTokenizer.from_pretrained",
                                     mock_autotokenizer_from_pretrained):
                res = tensor_search.search(config=self.config, text='hello', index_name=self.index_name_1)

        mock_automodel_from_pretrained.assert_called_once_with(
            public_repo_name, use_auth_token=None, cache_dir=ModelCache.hf_cache_path
        )

        mock_autotokenizer_from_pretrained.assert_called_once_with(
            public_repo_name, use_auth_token=None
        )

    def test_62_load_model_from_public_hf_repo_without_auth_using_name_add_documents(self):
        public_repo_name = "sentence-transformers/all-MiniLM-L6-v1"

        model_properties = {
            "name": public_repo_name,
            "dimensions": 384,
            "type": "hf",
        }
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1,
                                          index_settings=s3_settings)


        mock_automodel_from_pretrained = mock.MagicMock(side_effect=AutoModel.from_pretrained)

        with mock.patch('transformers.AutoModel.from_pretrained', new=mock_automodel_from_pretrained):
            tensor_search.add_documents(config=self.config, add_docs_params=AddDocsParams(
                index_name=self.index_name_1, auto_refresh=True, docs=[{'a': 'b'}]))

        mock_automodel_from_pretrained.assert_called_once_with(
            public_repo_name, use_auth_token=None, cache_dir=ModelCache.hf_cache_path
        )

    def test_hf_token_is_skipped_when_auth_required_is_False(self):
        public_repo_name = "sentence-transformers/all-MiniLM-L6-v1"
        hf_token = "some-secret-token"

        model_properties = {
            "dimensions": 384,
            "model_location": {
                "auth_required": False,
                "hf": {"repo_id": public_repo_name}
            },
            "type": "hf",
        }

        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1,
                                          index_settings=s3_settings)

        # Redirect the private model to a public one to finish the test
        mock_automodel_from_pretrained = mock.MagicMock()
        mock_automodel_from_pretrained.return_value = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v1")

        mock_autotokenizer_from_pretrained = mock.MagicMock()
        mock_autotokenizer_from_pretrained.return_value = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v1")

        with unittest.mock.patch("transformers.AutoModel.from_pretrained", mock_automodel_from_pretrained):
            with unittest.mock.patch("transformers.AutoTokenizer.from_pretrained",
                                     mock_autotokenizer_from_pretrained):
                res = tensor_search.search(config=self.config, text='hello', index_name=self.index_name_1,
                                           model_auth=ModelAuth(hf=HfAuth(token=hf_token)))

        mock_automodel_from_pretrained.assert_called_once_with(
            public_repo_name,
            use_auth_token=None,
            cache_dir = ModelCache.hf_cache_path
        )

        mock_autotokenizer_from_pretrained.assert_called_once_with(
            public_repo_name,
            use_auth_token=None,
        )

    def test_invalid_hf_location(self):
        private_repo_name = "your-private-repo"
        invalid_model_properties_list = [
        {
            "dimensions": 384,
            "model_location": {
                "auth_required": True,
                "hf": {"filename":"random again"}
            },
            "type": "hf",
        },
        {
            "dimensions": 384,
            "model_location": {
                "auth_required": True,
                "hf": {"name": 3,
                       "filename":"random again"}
            },
            "type": "hf",
        },
        ]
        for invalid_model_properties in invalid_model_properties_list:
            try:
                model_location = ModelLocation(**invalid_model_properties['model_location'])
                raise AssertionError
            except (InvalidArgError, ValidationError) as e:
                pass

    def test_load_model_from_private_hf_repo_with_redirect(self):
        private_repo_name = "your-private-repo"
        hf_token = "some-secret-token"

        redirect_hf_repo = "sentence-transformers/all-MiniLM-L6-v1"
        redirect_hf_token = None

        model_properties = {
            "dimensions": 384,
            "model_location": {
                "auth_required": True,
                "hf": {"repo_id": private_repo_name}
            },
            "type": "hf",
        }

        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1,
                                          index_settings=s3_settings)

        # Redirect the private model to a public one to finish the test
        mock_automodel_from_pretrained = mock.MagicMock()
        mock_automodel_from_pretrained.return_value = AutoModel.from_pretrained(
            redirect_hf_repo, use_auth_token=redirect_hf_token)

        mock_autotokenizer_from_pretrained = mock.MagicMock()
        mock_autotokenizer_from_pretrained.return_value = AutoTokenizer.from_pretrained(
            redirect_hf_repo, use_auth_token=redirect_hf_token)

        with unittest.mock.patch("transformers.AutoModel.from_pretrained", mock_automodel_from_pretrained):
            with unittest.mock.patch("transformers.AutoTokenizer.from_pretrained", mock_autotokenizer_from_pretrained):
                res = tensor_search.search(config=self.config, text='hello', index_name=self.index_name_1,
                                           model_auth=ModelAuth(hf=HfAuth(token=hf_token)))

        assert len(mock_automodel_from_pretrained.call_args_list) == 1
        assert mock_automodel_from_pretrained.call_args_list[0][0][0] == private_repo_name
        mock_autotokenizer_from_pretrained.call_args_list[0][1]["use_auth_token"] == hf_token

        assert len(mock_autotokenizer_from_pretrained.call_args_list) == 1
        assert mock_autotokenizer_from_pretrained.call_args_list[0][0][0] == private_repo_name
        mock_autotokenizer_from_pretrained.call_args_list[0][1]["use_auth_token"] == hf_token

class TestS3ModelAuthlLoadForHFModelVariants(MarqoTestCase):
    """
    This class tests the variants of the hf model loading and the expected error messages
    """

    device = 'cpu'

    def setUp(self) -> None:
        self.endpoint = self.authorized_url
        self.generic_header = {"Content-type": "application/json"}
        self.index_name_1 = "test-model-auth-index-1"
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass

    def tearDown(self) -> None:
        try:
            tensor_search.delete_index(config=self.config, index_name=self.index_name_1)
        except IndexNotFoundError as s:
            pass
        clear_loaded_models()


    def test_model_auth_mismatch_param_s3_ix(self):
        """This test is finished in open_clip test"""
        pass

    def test_model_loads_from_all_add_docs_derivatives(self):
        """Does it work from add_docs, add_docs orchestrator and add_documents_mp?
        """
        s3_object_key = 'path/to/your/secret_model.zip'
        s3_bucket = 'your-bucket-name'

        fake_access_key_id = '12345'
        fake_secret_key = 'this-is-a-secret'

        model_properties = {
            "dimensions": 384,
            "model_location": {
                "s3": {
                    "Bucket": s3_bucket,
                    "Key": s3_object_key,
                },
                "auth_required": True
            },
            "type": "hf",
        }

        model_abs_path = get_s3_model_absolute_cache_path(
            S3Location(
                Key=s3_object_key,
                Bucket=s3_bucket
            ), download_dir=ModelCache.hf_cache_path)

        _delete_file(model_abs_path)

        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)

        for add_docs_method, kwargs in [
                (tensor_search.add_documents_orchestrator, {'batch_size': 10}),
            ]:
            try:
                tensor_search.eject_model(model_name='my_model' ,device=self.device)
            except ModelNotInCacheError:
                pass
            # Create a mock Boto3 client
            mock_s3_client = mock.MagicMock()

            # Mock the generate_presigned_url method of the mock Boto3 client with a real OpenCLIP model, so that
            # the rest of the logic works.
            public_url = "https://marqo-cache-sentence-transformers.s3.us-west-2.amazonaws.com/all-MiniLM-L6-v1/all-MiniLM-L6-v2.zip"
            mock_s3_client.generate_presigned_url.return_value = public_url

            with unittest.mock.patch('boto3.client', return_value=mock_s3_client) as mock_boto3_client:
                with self.assertRaises(BadRequestError) as cm:
                    with unittest.mock.patch(
                        'marqo.s2_inference.processing.custom_clip_utils.download_pretrained_from_url'
                    ) as mock_download_pretrained_from_url:
                        add_docs_method(
                            config=self.config,
                            add_docs_params=AddDocsParams(
                                index_name=self.index_name_1,
                                model_auth=ModelAuth(s3=S3Auth(
                                    aws_access_key_id=fake_access_key_id,
                                    aws_secret_access_key=fake_secret_key)),
                                auto_refresh=True,
                                docs=[{f'Title': "something {i} good"} for i in range(20)]
                            ),
                            **kwargs
                        )
            mock_download_pretrained_from_url.assert_called_once_with(
                url=public_url, cache_dir=ModelCache.hf_cache_path, cache_file_name='secret_model.zip')
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
            mock_download_pretrained_from_url.reset_mock()
            mock_s3_client.reset_mock()
            mock_boto3_client.reset_mock()

    def test_model_loads_from_multi_search(self):
        s3_object_key = 'path/to/your/secret_model.pt'
        s3_bucket = 'your-bucket-name'

        fake_access_key_id = '12345'
        fake_secret_key = 'this-is-a-secret'

        model_properties = {
            "dimensions": 384,
            "model_location": {
                "s3": {
                    "Bucket": s3_bucket,
                    "Key": s3_object_key,
                },
                "auth_required": True
            },
            "type": "hf",
        }
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)

        try:
            tensor_search.eject_model(model_name='my_model', device=self.device)
        except ModelNotInCacheError:
            pass


        with unittest.mock.patch('marqo.s2_inference.s2_inference.vectorise',
                                 side_effect=fake_vectorise_384) as mock_vectorise:
            model_auth = ModelAuth(
                s3=S3Auth(
                    aws_access_key_id=fake_access_key_id,
                    aws_secret_access_key=fake_secret_key)
            )
            res = tensor_search.search(
                index_name=self.index_name_1,
                config=self.config,
                model_auth=model_auth,
                text={
                    (f"https://raw.githubusercontent.com/marqo-ai/"
                     f"marqo-api-tests/mainline/assets/ai_hippo_realistic.png"): 0.3,
                    'my text': -1.3
                },
            )
            assert 'hits' in res
            mock_vectorise.assert_called()
            assert len(mock_vectorise.call_args_list) > 0
            for _args, _kwargs in mock_vectorise.call_args_list:
                assert _kwargs['model_properties']['model_location'] == {
                    "s3": {
                        "Bucket": s3_bucket,
                        "Key": s3_object_key,
                    },
                    "auth_required": True
                }
                assert _kwargs['model_auth'] == model_auth

    def test_model_loads_from_multimodal_combination(self):
        s3_object_key = 'path/to/your/secret_model.pt'
        s3_bucket = 'your-bucket-name'

        fake_access_key_id = '12345'
        fake_secret_key = 'this-is-a-secret'

        model_properties = {
            "dimensions": 384,
            "model_location": {
                "s3": {
                    "Bucket": s3_bucket,
                    "Key": s3_object_key,
                },
                "auth_required": True
            },
            "type": "hf",
        }
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)

        for add_docs_method, kwargs in [
            (tensor_search.add_documents_orchestrator, {'batch_size': 10}),
            (tensor_search.add_documents, {})
        ]:
            try:
                tensor_search.eject_model(model_name='my_model', device=self.device)
            except ModelNotInCacheError:
                pass
            # Create a mock Boto3 client
            mock_s3_client = mock.MagicMock()

            mock_s3_client.generate_presigned_url.return_value = "https://random_model.zip"

            with unittest.mock.patch('marqo.s2_inference.s2_inference.vectorise', side_effect=fake_vectorise_384) as mock_vectorise:
                model_auth = ModelAuth(
                    s3=S3Auth(
                    aws_access_key_id=fake_access_key_id,
                    aws_secret_access_key=fake_secret_key)
                )
                res = add_docs_method(
                    config=self.config,
                    add_docs_params=AddDocsParams(
                        index_name=self.index_name_1,
                        model_auth=model_auth,
                        auto_refresh=True,
                        docs=[{
                            'my_combination_field': {
                                'my_image': f"https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
                                'some_text': f"my text {i}"}} for i in range(20)],
                        mappings={
                            "my_combination_field": {
                                "type": "multimodal_combination",
                                "weights": {
                                    "my_image": 0.5,
                                    "some_text": 0.5
                                }
                            }
                        }
                    ),
                    **kwargs
                )
                if isinstance(res, list):
                    assert all([not batch_res ['errors'] for batch_res in res])
                else:
                    assert not res['errors']
                mock_vectorise.assert_called()
                for _args, _kwargs in mock_vectorise.call_args_list:
                    assert _kwargs['model_properties']['model_location'] == {
                        "s3": {
                            "Bucket": s3_bucket,
                            "Key": s3_object_key,
                        },
                        "auth_required": True
                    }
                    assert _kwargs['model_auth'] == model_auth

    def test_no_creds_error(self):
        """in s3, if there aren't creds"""
        s3_object_key = 'path/to/your/secret_model.pt'
        s3_bucket = 'your-bucket-name'

        model_properties = {
            "dimensions": 384,
            "model_location": {
                "s3": {
                    "Bucket": s3_bucket,
                    "Key": s3_object_key,
                },
                "auth_required": True
            },
            "type": "hf",
        }
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)



        # Create a mock Boto3 client
        mock_s3_client = mock.MagicMock()

        mock_s3_client.generate_presigned_url.return_value = "https://dummy_model.zip"

        with unittest.mock.patch('boto3.client', return_value=mock_s3_client):
            with self.assertRaises(BadRequestError) as cm:
                tensor_search.search(
                    config=self.config, text='hello', index_name=self.index_name_1,
                )
                self.assertIn("s3 authorisation information is required", str(cm.exception))


        with unittest.mock.patch('boto3.client', return_value=mock_s3_client):
            with self.assertRaises(BadRequestError) as cm2:
                res = tensor_search.add_documents(
                    config=self.config, add_docs_params=AddDocsParams(
                        index_name=self.index_name_1, auto_refresh=True,
                        docs=[{'title': 'blah blah'}]
                    )
                )
            self.assertIn("s3 authorisation information is required", str(cm2.exception))

    def test_bad_creds_error_s3(self):
        """in s3 if creds aren't valid. Ensure a helpful error"""
        s3_object_key = 'path/to/your/secret_model.pt'
        s3_bucket = 'your-bucket-name'

        fake_access_key_id = '12345'
        fake_secret_key = 'this-is-a-secret'

        model_auth = ModelAuth(
            s3=S3Auth(
                aws_access_key_id=fake_access_key_id,
                aws_secret_access_key=fake_secret_key)
        )

        model_properties = {
            "dimensions": 384,
            "model_location": {
                "s3": {
                    "Bucket": s3_bucket,
                    "Key": s3_object_key,
                },
                "auth_required": True
            },
            "type": "hf",
        }
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)

        with self.assertRaises(BadRequestError) as cm:
            tensor_search.search(
                config=self.config, text='hello', index_name=self.index_name_1,
                model_auth=model_auth
            )
        self.assertIn("403 error when trying to retrieve model from s3", str(cm.exception))

        with self.assertRaises(BadRequestError) as cm2:
            res = tensor_search.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.index_name_1, auto_refresh=True,
                    docs=[{'title': 'blah blah'}], model_auth=model_auth
                )
            )
        self.assertIn("403 error when trying to retrieve model from s3", str(cm2.exception))

    def test_non_existent_hf_location(self):
        hf_object = "some_model.pt"
        hf_repo_name = "MyRepo/test-private"
        hf_token = "hf_some_secret_key"

        model_auth = ModelAuth(
            hf=HfAuth(token=hf_token)
        )

        model_properties = {
            "dimensions": 384,
            "model_location": {
                "hf": {
                    "repo_id": hf_repo_name,
                    "filename": hf_object,
                },
                "auth_required": True
            },
            "type": "hf",
        }
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)

        with self.assertRaises(BadRequestError) as cm:
            tensor_search.search(
                config=self.config, text='hello', index_name=self.index_name_1,
                model_auth=model_auth
            )

        self.assertIn("Could not find the specified Hugging Face model repository.", str(cm.exception))

        with self.assertRaises(BadRequestError) as cm2:
            res = tensor_search.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.index_name_1, auto_refresh=True,
                    docs=[{'title': 'blah blah'}], model_auth=model_auth
                )
            )
        self.assertIn("Could not find the specified Hugging Face model repository.", str(cm.exception))

    def test_bad_creds_error_hf(self):
        """the model and repo do exist, but creds are bad. raises the same type of error
        as the previous one. """
        hf_object = "dummy_model.pt"
        hf_repo_name = "Marqo/test-private"
        hf_token = "hf_some_secret_key"

        model_auth = ModelAuth(
            hf=HfAuth(token=hf_token)
        )

        model_properties = {
            "dimensions": 384,
            "model_location": {
                "hf": {
                    "repo_id": hf_repo_name,
                    "filename": hf_object,
                },
                "auth_required": True
            },
            "type": "hf",
        }
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)

        with self.assertRaises(BadRequestError) as cm:
            tensor_search.search(
                config=self.config, text='hello', index_name=self.index_name_1,
                model_auth=model_auth
            )
        self.assertIn("Could not find the specified Hugging Face model repository.", str(cm.exception))

        with self.assertRaises(BadRequestError) as cm2:
            res = tensor_search.add_documents(
                config=self.config, add_docs_params=AddDocsParams(
                    index_name=self.index_name_1, auto_refresh=True,
                    docs=[{'title': 'blah blah'}], model_auth=model_auth
                )
            )
        self.assertIn("Could not find the specified Hugging Face model repository.", str(cm.exception))

    def test_bulk_search(self):
        """Does it work with bulk search, including multi search
        """
        s3_object_key = 'path/to/your/secret_model.zip'
        s3_bucket = 'your-bucket-name'

        fake_access_key_id = '12345'
        fake_secret_key = 'this-is-a-secret'

        model_auth = ModelAuth(
            s3=S3Auth(
                aws_access_key_id=fake_access_key_id,
                aws_secret_access_key=fake_secret_key)
        )

        model_properties = {
            "dimensions": 384,
            "model_location": {
                "s3": {
                    "Bucket": s3_bucket,
                    "Key": s3_object_key,
                },
                "auth_required": True
            },
            "type": "hf",
        }
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)

        for bulk_search_query in [
                BulkSearchQuery(queries=[
                    BulkSearchQueryEntity(
                        index=self.index_name_1, q="match", searchMethod="TENSOR",
                        modelAuth=model_auth
                    ),
                    BulkSearchQueryEntity(
                        index=self.index_name_1, q={"random text": 0.5, "other_text": -0.3},
                        searchableAttributes=["abc"], searchMethod="TENSOR",
                        modelAuth=model_auth
                    ),
                ]),
                BulkSearchQuery(queries=[
                    BulkSearchQueryEntity(
                        index=self.index_name_1, q={"random text": 0.5, "other_text": -0.3},
                        searchableAttributes=["abc"], searchMethod="TENSOR",
                        modelAuth=model_auth
                    ),
                ])
            ]:
            try:
                tensor_search.eject_model(model_name='my_model' ,device=self.device)
            except ModelNotInCacheError:
                pass
            # Create a mock Boto3 client
            mock_s3_client = mock.MagicMock()

            mock_s3_client.generate_presigned_url.return_value = "https://some_non_existent_model.pt"

            with unittest.mock.patch('boto3.client', return_value=mock_s3_client) as mock_boto3_client:
                with self.assertRaises(InvalidArgError) as cm:
                    with unittest.mock.patch(
                        'marqo.s2_inference.processing.custom_clip_utils.download_pretrained_from_url'
                    ) as mock_download_pretrained_from_url:
                        tensor_search.bulk_search(
                            query=bulk_search_query,
                            marqo_config=self.config,
                        )
            mock_download_pretrained_from_url.assert_called_once_with(
                url='https://some_non_existent_model.pt', cache_dir=ModelCache.hf_cache_path, cache_file_name='secret_model.zip')
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

            mock_download_pretrained_from_url.reset_mock()
            mock_s3_client.reset_mock()
            mock_boto3_client.reset_mock()

    def test_bulk_search_vectorise(self):
        """are the calls to vectorise expected? work with bulk search, including multi search
        """
        s3_object_key = 'path/to/your/secret_model.pt'
        s3_bucket = 'your-bucket-name'

        fake_access_key_id = '12345'
        fake_secret_key = 'this-is-a-secret'

        model_auth = ModelAuth(
            s3=S3Auth(
                aws_access_key_id=fake_access_key_id,
                aws_secret_access_key=fake_secret_key)
        )

        model_properties = {
            "dimensions": 384,
            "model_location": {
                "s3": {
                    "Bucket": s3_bucket,
                    "Key": s3_object_key,
                },
                "auth_required": True
            },
            "type": "hf",
        }
        s3_settings = _get_base_index_settings()
        s3_settings['index_defaults']['model_properties'] = model_properties
        tensor_search.create_vector_index(config=self.config, index_name=self.index_name_1, index_settings=s3_settings)

        for bulk_search_query in [
                BulkSearchQuery(queries=[
                    BulkSearchQueryEntity(
                        index=self.index_name_1, q="match", searchMethod="TENSOR",
                        modelAuth=model_auth
                    ),
                    BulkSearchQueryEntity(
                        index=self.index_name_1, q={"random text": 0.5, "other_text": -0.3},
                        searchableAttributes=["abc"], searchMethod="TENSOR",
                        modelAuth=model_auth
                    ),
                ]),
                BulkSearchQuery(queries=[
                    BulkSearchQueryEntity(
                        index=self.index_name_1, q={"random text": 0.5, "other_text": -0.3},
                        searchableAttributes=["abc"], searchMethod="TENSOR",
                        modelAuth=model_auth
                    ),
                ])
            ]:
            try:
                tensor_search.eject_model(model_name='my_model' ,device=self.device)
            except ModelNotInCacheError:
                pass
            # Create a mock Boto3 client
            mock_s3_client = mock.MagicMock()

            # Mock the generate_presigned_url method of the mock Boto3 client with a real OpenCLIP model, so that
            # the rest of the logic works.
            mock_s3_client.generate_presigned_url.return_value = "https://some_non_existent_model.pt"

            with unittest.mock.patch('marqo.s2_inference.s2_inference.vectorise',
                                     side_effect=fake_vectorise) as mock_vectorise:
                        tensor_search.bulk_search(
                            query=bulk_search_query,
                            marqo_config=self.config,
                        )
            mock_vectorise.assert_called()
            for _args, _kwargs in mock_vectorise.call_args_list:
                assert _kwargs['model_properties']['model_location'] == {
                    "s3": {
                        "Bucket": s3_bucket,
                        "Key": s3_object_key,
                    },
                    "auth_required": True
                }
                assert _kwargs['model_auth'] == model_auth

            mock_vectorise.reset_mock()

















