from unittest import TestCase
from unittest.mock import patch, MagicMock

import pytest

from marqo.inference.native_inference.embedding_models.open_clip_model import OpenCLIPModel
from marqo.s2_inference.configs import ModelCache
from marqo.s2_inference.errors import InvalidModelPropertiesError
from marqo.s2_inference.model_registry import _get_open_clip_properties
from marqo.tensor_search.models.external_apis.s3 import S3Auth
from marqo.tensor_search.models.private_models import ModelAuth, ModelLocation


class TestOpenCLIPModelLoad(TestCase):
    """A test suite for loading OpenCLIP models.

    The model loading logic for OpenCLIP models in Marqo can be categorized into the following steps in order of:
    1. If the `url` or `modelLocation` is provided in the model properties, download the model from the specified
    location and load the checkpoint.
    2. If the `name` of model properties is provided, and it starts with `hf-hub`, load the model from the Hugging Face.
    3. Otherwise, load the model as a registered model in the model registry.
    """

    def test_load_OpenCLIPModelFromCheckPointMethod_success(self):
        """Test loading an OpenCLIP model from a checkpoint is called when providing a url in the model properties."""
        model_name = "my_test_model"
        model_properties = {
            "name": "ViT-B-32",
            "url": "https://openclipart.org/download/12345/my_test_model.pt",
            "type": "open_clip",
            "dimensions": 512
        }

        with patch("marqo.inference.native_inference.embedding_models.open_clip_model.OpenCLIPModel._load_model_and_image_preprocessor_from_checkpoint", \
                   return_value=(MagicMock(), MagicMock())) as mock_load_method:
            with patch("marqo.inference.native_inference.embedding_models.open_clip_model.OpenCLIPModel._load_tokenizer_from_checkpoint",
                       return_value=MagicMock()) as mock_load_tokenizer:
                with patch.object(MagicMock(), 'eval', return_value=None) as mock_eval:
                    model = OpenCLIPModel(model_properties=model_properties, device="cpu")
                    model.load()
                    mock_load_method.assert_called_once()
                    mock_load_tokenizer.assert_called_once()

    def test_load_OpenCLIPModelFromCheckPointParameters_success(self):
        """Test correct parameters are passed to the OpenCLIP loading from checkpoint method."""
        model_tag = "my_test_model"
        model_properties = {
            "name": "ViT-B-108",
            "url": "https://openclipart.org/download/12345/my_test_model.pt",
            "type": "open_clip",
            "dimensions": 512
        }
        with patch("marqo.inference.native_inference.embedding_models.open_clip_model.open_clip.create_model", return_value=MagicMock()) \
                as mock_create_model:
            with patch("marqo.inference.native_inference.embedding_models.open_clip_model.open_clip.get_tokenizer", return_value=MagicMock()) \
                    as mock_tokenizer:
                with patch("marqo.inference.native_inference.embedding_models.open_clip_model.download_model", return_value="my_test_model.pt"):
                    with patch.object(MagicMock(), 'eval', return_value=None) as mock_eval:
                        model = OpenCLIPModel(model_properties=model_properties, device="cpu")
                        model.load()
                        mock_create_model.assert_called_once_with(
                            model_name="ViT-B-108",
                            jit=False,
                            pretrained="my_test_model.pt",
                            precision="fp32", device="cpu",
                            cache_dir=ModelCache.clip_cache_path
                        )
                        mock_tokenizer.assert_called_once_with("ViT-B-108")
                        preprocess_config = model.image_preprocessor_config
                        self.assertEqual(224, preprocess_config.size)
                        self.assertEqual("RGB", preprocess_config.mode)
                        self.assertEqual((0.48145466, 0.4578275, 0.40821073), preprocess_config.mean)
                        self.assertEqual((0.26862954, 0.26130258, 0.27577711), preprocess_config.std)
                        self.assertEqual("bicubic", preprocess_config.interpolation)
                        self.assertEqual("shortest", preprocess_config.resize_mode)
                        self.assertEqual(0, preprocess_config.fill_color)

    def test_load_OpenCLIPModelFromCheckPointPreprocessConfig(self):
        """Test correct parameters are passed to the OpenCLIP loading from checkpoint method."""
        model_tag = "my_test_model"
        model_properties = {
            "dimensions": 512,
            "name": "test-siglip",
            "url": "https://openclipart.org/download/12345/my_test_model.pt",
            "type": "open_clip",
            "image_preprocessor": "SigLIP",
            "size": 322  # Override the default size 224
        }
        with patch("marqo.inference.native_inference.embedding_models.open_clip_model.open_clip.create_model", return_value=MagicMock()) \
                as mock_create_model:
            with patch("marqo.inference.native_inference.embedding_models.open_clip_model.open_clip.get_tokenizer", return_value=MagicMock()) \
                    as mock_tokenizer:
                with patch("marqo.inference.native_inference.embedding_models.open_clip_model.download_model", return_value="my_test_model.pt"):
                    with patch.object(MagicMock(), 'eval', return_value=None) as mock_eval:
                        model = OpenCLIPModel(model_properties=model_properties, device="cpu")
                        model.load()
                        mock_create_model.assert_called_once_with(
                            model_name="test-siglip",
                            jit=False,
                            pretrained="my_test_model.pt",
                            precision="fp32", device="cpu",
                            cache_dir=ModelCache.clip_cache_path
                        )
                        mock_tokenizer.assert_called_once_with("test-siglip")
                        preprocess_config = model.image_preprocessor_config
                        self.assertEqual(322, preprocess_config.size)
                        self.assertEqual("RGB", preprocess_config.mode)
                        self.assertEqual((0.5, 0.5, 0.5), preprocess_config.mean)
                        self.assertEqual((0.5, 0.5, 0.5), preprocess_config.std)
                        self.assertEqual("bicubic", preprocess_config.interpolation)
                        self.assertEqual("squash", preprocess_config.resize_mode)
                        self.assertEqual(0, preprocess_config.fill_color)

    def test_open_clip_load_fromHuggingFaceHub_success(self):
        model_tag = "my_test_model"
        model_properties = {
            "name": "hf-hub:my_test_hub",
            "type": "open_clip",
            "dimensions": 512
        }
        with patch("marqo.s2_inference.clip_utils.open_clip.create_model_and_transforms",
                   return_value=(MagicMock(), MagicMock(), MagicMock())) \
                as mock_create_model:
            with patch("marqo.s2_inference.clip_utils.open_clip.get_tokenizer", return_value=MagicMock()) \
                    as mock_tokenizer:
                with patch.object(MagicMock(), 'eval', return_value=None) as mock_eval:
                    model = OpenCLIPModel(model_properties=model_properties, device="cpu")
                    model.load()
                    mock_create_model.assert_called_once_with(
                        model_name="hf-hub:my_test_hub",
                        device="cpu",
                        cache_dir=ModelCache.clip_cache_path
                    )
                    mock_tokenizer.assert_called_once_with("hf-hub:my_test_hub")

    def test_open_clip_load_fromMarqoModelRegistry_success(self):
        model_tag = "open_clip/ViT-B-32/laion5b_s13b_b90k"
        model_properties = {
            "name": "open_clip/ViT-B-32/laion5b_s13b_b90k",
            "type": "open_clip",
            "dimensions": 512
        }
        with patch("marqo.s2_inference.clip_utils.open_clip.create_model_and_transforms",
                   return_value=(MagicMock(), MagicMock(), MagicMock())) \
                as mock_create_model:
            with patch("marqo.s2_inference.clip_utils.open_clip.get_tokenizer", return_value=MagicMock()) \
                    as mock_tokenizer:
                with patch.object(MagicMock(), 'eval', return_value=None) as mock_eval:
                    model = OpenCLIPModel(model_properties=model_properties, device="cpu")
                    model.load()
                    mock_create_model.assert_called_once_with(
                        model_name="ViT-B-32",
                        pretrained="laion5b_s13b_b90k",
                        device="cpu",
                        cache_dir=ModelCache.clip_cache_path
                    )
                    mock_tokenizer.assert_called_once_with("ViT-B-32")

    def test_load_OpenCLIPModel_missing_model_properties(self):
        """Test loading an OpenCLIP model with missing model properties should raise an error."""
        model_tag = "my_test_model"
        model_properties = {
            "type": "open_clip",
            "dimensions": 512
            # Missing 'name' and 'url'
        }

        with self.assertRaises(InvalidModelPropertiesError) as context:
            model = OpenCLIPModel(model_properties=model_properties, device="cpu")
            model.load()

        self.assertIn("validation error", str(context.exception))
        self.assertIn("name", str(context.exception))

    def test_load_OpenCLIPModel_unsupported_image_preprocessor(self):
        """Test loading an OpenCLIP model with an unsupported image preprocessor should raise an error."""
        model_tag = "my_test_model"
        model_properties = {
            "name": "ViT-B-32",
            "type": "open_clip",
            "image_preprocessor": "UnsupportedPreprocessor",
            "dimensions": 512
        }

        with self.assertRaises(InvalidModelPropertiesError) as context:
            model = OpenCLIPModel(model_properties=model_properties, device="cpu")
            model.load()

        self.assertIn("permitted: 'SigLIP', 'OpenAI', 'OpenCLIP', 'CLIPA'", str(context.exception))

    def test_load_OpenCLIPModel_from_local_path(self):
        """Test loading an OpenCLIP model from a local path."""
        model_tag = "my_test_model"
        model_properties = {
            "name": "ViT-B-32",
            "localpath": "/path/to/my_test_model.pt",
            "dimensions": 512,
            "type": "open_clip"
        }
        with patch("marqo.inference.native_inference.embedding_models.open_clip_model.open_clip.create_model", return_value=MagicMock()) \
                as mock_create_model:
            with patch("marqo.inference.native_inference.embedding_models.open_clip_model.open_clip.get_tokenizer", return_value=MagicMock()) \
                    as mock_tokenizer:
                with patch.object(MagicMock(), 'eval', return_value=None) as mock_eval:
                    with patch("marqo.inference.native_inference.embedding_models.open_clip_model.os.path.exists",
                               return_value=True) as mock_path_exists:
                        model = OpenCLIPModel(model_properties=model_properties, device="cpu")
                        model.load()
                        mock_create_model.assert_called_once_with(
                            model_name="ViT-B-32",
                            jit=False,
                            pretrained="/path/to/my_test_model.pt",
                            precision="fp32", device="cpu",
                            cache_dir=ModelCache.clip_cache_path
                        )
                        mock_tokenizer.assert_called_once_with("ViT-B-32")
                        mock_path_exists.assert_called_once_with("/path/to/my_test_model.pt")

    def test_load_OpenCLIPModel_with_auth_s3(self):
        """Ensure that the model/checkpoint is downloaded with the correct S3 authentication."""
        model_tag = "my_test_model"
        model_properties = {
            "name": "ViT-B-16",
            "model_location": {
                "s3": {
                    "Bucket": "my-bucket",
                    "Key": "my-key",
                },
                "authRequired": True,
            },
            "type": "open_clip",
            "dimensions": 768,
        }

        model_auth = ModelAuth(s3 = S3Auth(
            aws_access_key_id="my_access_key",
            aws_secret_access_key="my_secret_key",
        ))

        with patch("marqo.inference.native_inference.embedding_models.open_clip_model.download_model") as mock_download_model:
            # It's ok to return a RuntimeError as we are testing the download_model function
            with self.assertRaises(RuntimeError):
                model = OpenCLIPModel(model_properties=model_properties, device="cpu", model_auth=model_auth)
                model.load()

            mock_download_model.assert_called_once_with(
                repo_location=ModelLocation(**model_properties["model_location"]),
                auth=model_auth,
            )

    def test_load_OpenCLIPModel_with_auth_hf(self):
        """Ensure that the model/checkpoint is downloaded with the correct S3 authentication."""
        model_tag = "my_test_model"
        model_properties = {
            "name": "ViT-B-16",
            "model_location": {
                "hf": {
                    "repo_id": "my-hf-repo",
                    "filename": "my-hf-filename.pt"
                },
                "authRequired": True,
            },
            "type": "open_clip",
            "dimensions": 768
        }

        model_auth = ModelAuth(**{"hf": {"token":"my_hf_token"}})

        with patch("marqo.inference.native_inference.embedding_models.open_clip_model.download_model") as mock_download_model:
            # It's ok to return a RuntimeError as we are testing the download_model function
            with self.assertRaises(RuntimeError) as e:
                model = OpenCLIPModel(model_properties=model_properties, device="cpu", model_auth=model_auth)
                model.load()

            mock_download_model.assert_called_once_with(
                repo_location=ModelLocation(**model_properties["model_location"]),
                auth=model_auth,
            )

    def test_load_legacy_openai_clip_model(self):
        """A test to ensure old OpenAI CLIP models (e.g., ViT-B/32) are loaded correctly."""
        model_properties = {
            "name": "ViT-B/32", # Old OpenAI CLIP model name
            "type": "open_clip",
            "url": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt",
            "dimensions": 512
        }
        model = OpenCLIPModel(model_properties=model_properties, device="cpu")
        model.load()