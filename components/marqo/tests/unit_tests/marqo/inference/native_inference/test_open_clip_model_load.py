from unittest import TestCase
from unittest.mock import patch, MagicMock

from marqo.inference.native_inference.embedding_models.open_clip_model import OpenCLIPModel
from marqo.s2_inference.configs import ModelCache


class TestOpenCLIPModelLoad(TestCase):
    """A test suite for loading OpenCLIP models.

    The model loading logic for OpenCLIP models in Marqo can be categorized into the following steps in order of:
    1. If the `url` or `modelLocation` is provided in the model properties, download the model from the specified
    location and load the checkpoint.
    2. If the `name` of model properties is provided, and it starts with `hf-hub`, load the model from the Hugging Face.
    3. Otherwise, load the model as a registered model in the model registry.
    """

    def test_the_string_replace_for_legacy_model_does_not_apply_to_hf(self):
        """
        A test to ensure the string replace ("/", "-") for legacy model names does not apply to Hugging Face models.
        """
        model_properties = {
            "name": "hf-hub:timm/ViT-B-16-SigLIP",
            "type": "open_clip",
            "dimensions": 768,
            "url": "https://huggingface.co/Marqo/marqo-fashionSigLIP/resolve/main/open_clip_pytorch_model.bin",
            "imagePreprocessor": "SigLIP"
        }
        with patch("marqo.inference.native_inference.embedding_models.open_clip_model.open_clip.create_model",
                   return_value=MagicMock()) as mock_create_model, \
            patch("marqo.inference.native_inference.embedding_models.open_clip_model.open_clip.get_tokenizer",
                   return_value=MagicMock()) as mock_tokenizer, \
            patch("marqo.inference.native_inference.embedding_models.open_clip_model.download_model",
                  return_value="my_test_model.pt"), \
            patch.object(MagicMock(), 'eval', return_value=None) as mock_eval:

            model = OpenCLIPModel(model_properties=model_properties, device="cpu")
            model.load()
            mock_create_model.assert_called_once_with(
                model_name="hf-hub:timm/ViT-B-16-SigLIP",
                jit=False,
                pretrained="my_test_model.pt",
                precision="fp32",
                device="cpu",
                cache_dir=ModelCache.clip_cache_path
            )
            # Ensure the name is not modified
            mock_tokenizer.assert_called_once_with("hf-hub:timm/ViT-B-16-SigLIP")

    def test_the_string_replace_for_legacy_model_apply_to_legacy_models(self):
        """
        A test to ensure the string replace ("/", "-") for legacy model names apply to Hugging Face models.
        """

        model_properties = {
            "name": "ViT-L/14",
            "type": "open_clip",
            "url": "https://a-dummy-url/clip_vit_l_14.pt",
            "dimensions": 768,
        }
        with patch("marqo.inference.native_inference.embedding_models.open_clip_model.open_clip.create_model",
                   return_value=MagicMock()) as mock_create_model, \
            patch("marqo.inference.native_inference.embedding_models.open_clip_model.open_clip.get_tokenizer",
                   return_value=MagicMock()) as mock_tokenizer, \
            patch("marqo.inference.native_inference.embedding_models.open_clip_model.download_model",
                  return_value="my_test_model.pt"), \
            patch.object(MagicMock(), 'eval', return_value=None) as mock_eval:

            model = OpenCLIPModel(model_properties=model_properties, device="cpu")
            model.load()
            mock_create_model.assert_called_once_with(
                # This remains unchanged and open_clip.create_model will handle the replacement
                model_name="ViT-L/14",
                jit=False,
                pretrained="my_test_model.pt",
                precision="fp32",
                device="cpu",
                cache_dir=ModelCache.clip_cache_path
            )
            # Ensure the name is not modified
            mock_tokenizer.assert_called_once_with("ViT-L-14") # The name should be modified to "ViT-L-14"
