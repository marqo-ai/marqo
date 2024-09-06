from unittest import TestCase
from unittest.mock import patch, MagicMock

from marqo.s2_inference.clip_utils import OPEN_CLIP
from marqo.s2_inference.configs import ModelCache
from marqo.s2_inference.model_registry import _get_open_clip_properties

OPEN_CLIP_MODEL_PROPERTIES = _get_open_clip_properties()


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
            "type": "open_clip"
        }

        with patch("marqo.s2_inference.clip_utils.OPEN_CLIP._load_model_and_image_preprocessor_from_checkpoint", \
                   return_value=(MagicMock(), MagicMock())) as mock_load_method:
            with patch("marqo.s2_inference.clip_utils.OPEN_CLIP._load_tokenizer_from_checkpoint",
                       return_value=MagicMock()) as mock_load_tokenizer:
                with patch.object(MagicMock(), 'eval', return_value=None) as mock_eval:
                    model = OPEN_CLIP(model_properties=model_properties, device="cpu")
                    model.load()
                    mock_load_method.assert_called_once()
                    mock_load_tokenizer.assert_called_once()

    def test_load_OpenCLIPModelFromCheckPointParameters_success(self):
        """Test correct parameters are passed to the OpenCLIP loading from checkpoint method."""
        model_tag = "my_test_model"
        model_properties = {
            "name": "ViT-B-108",
            "url": "https://openclipart.org/download/12345/my_test_model.pt",
            "type": "open_clip"
        }
        with patch("marqo.s2_inference.clip_utils.open_clip.create_model", return_value=MagicMock()) \
                as mock_create_model:
            with patch("marqo.s2_inference.clip_utils.open_clip.get_tokenizer", return_value=MagicMock()) \
                    as mock_tokenizer:
                with patch("marqo.s2_inference.clip_utils.download_model", return_value="my_test_model.pt"):
                    with patch.object(MagicMock(), 'eval', return_value=None) as mock_eval:
                        model = OPEN_CLIP(model_properties=model_properties, device="cpu")
                        model.load()
                        mock_create_model.assert_called_once_with(
                            model_name="ViT-B-108",
                            jit=False,
                            pretrained="my_test_model.pt",
                            precision="fp32", device="cpu",
                            cache_dir=ModelCache.clip_cache_path
                        )
                        mock_tokenizer.assert_called_once_with("ViT-B-108")
                        preprocess_config = model.preprocess_config
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
            "name": "test-siglip",
            "url": "https://openclipart.org/download/12345/my_test_model.pt",
            "type": "open_clip",
            "image_preprocessor": "SigLIP",
            "size": 322  # Override the default size 224
        }
        with patch("marqo.s2_inference.clip_utils.open_clip.create_model", return_value=MagicMock()) \
                as mock_create_model:
            with patch("marqo.s2_inference.clip_utils.open_clip.get_tokenizer", return_value=MagicMock()) \
                    as mock_tokenizer:
                with patch("marqo.s2_inference.clip_utils.download_model", return_value="my_test_model.pt"):
                    with patch.object(MagicMock(), 'eval', return_value=None) as mock_eval:
                        model = OPEN_CLIP(model_properties=model_properties, device="cpu")
                        model.load()
                        mock_create_model.assert_called_once_with(
                            model_name="test-siglip",
                            jit=False,
                            pretrained="my_test_model.pt",
                            precision="fp32", device="cpu",
                            cache_dir=ModelCache.clip_cache_path
                        )
                        mock_tokenizer.assert_called_once_with("test-siglip")
                        preprocess_config = model.preprocess_config
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
        }
        with patch("marqo.s2_inference.clip_utils.open_clip.create_model_and_transforms",
                   return_value=(MagicMock(), MagicMock(), MagicMock())) \
                as mock_create_model:
            with patch("marqo.s2_inference.clip_utils.open_clip.get_tokenizer", return_value=MagicMock()) \
                    as mock_tokenizer:
                with patch.object(MagicMock(), 'eval', return_value=None) as mock_eval:
                    model = OPEN_CLIP(model_properties=model_properties, device="cpu")
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
        }
        with patch("marqo.s2_inference.clip_utils.open_clip.create_model_and_transforms",
                   return_value=(MagicMock(), MagicMock(), MagicMock())) \
                as mock_create_model:
            with patch("marqo.s2_inference.clip_utils.open_clip.get_tokenizer", return_value=MagicMock()) \
                    as mock_tokenizer:
                with patch.object(MagicMock(), 'eval', return_value=None) as mock_eval:
                    model = OPEN_CLIP(model_properties=model_properties, device="cpu")
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
            "type": "open_clip"
            # Missing 'name' and 'url'
        }

        with self.assertRaises(ValueError) as context:
            model = OPEN_CLIP(model_properties=model_properties, device="cpu")
            model.load()

        self.assertIn("validation error", str(context.exception))
        self.assertIn("name", str(context.exception))

    def test_load_OpenCLIPModel_unsupported_image_preprocessor(self):
        """Test loading an OpenCLIP model with an unsupported image preprocessor should raise an error."""
        model_tag = "my_test_model"
        model_properties = {
            "name": "ViT-B-32",
            "type": "open_clip",
            "image_preprocessor": "UnsupportedPreprocessor"
        }

        with self.assertRaises(ValueError) as context:
            model = OPEN_CLIP(model_properties=model_properties, device="cpu")
            model.load()

        self.assertIn("permitted: 'SigLIP', 'OpenAI', 'OpenCLIP', 'MobileCLIP', 'CLIPA'", str(context.exception))
