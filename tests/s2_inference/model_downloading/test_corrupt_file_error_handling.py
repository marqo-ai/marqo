import unittest
from unittest.mock import MagicMock, patch
from marqo.s2_inference.errors import InvalidModelPropertiesError
from marqo.s2_inference.s2_inference import _load_model
from marqo.s2_inference.hf_utils import extract_huggingface_archive
from marqo.s2_inference.configs import ModelCache
import zipfile
import tarfile

class TestCorruptFileInOpenCLIP(unittest.TestCase):
    '''Test that a corrupt file in OpenCLIP is handled correctly.
        Note that the downloaded file should be a single .bin, .pt file.
    '''

    def setUp(self):
        self.load_parameters = {
            "model_name": "test-corrupted-open-clip-model",
            "device": "cpu",
            "model_auth": None,
            "calling_func" : "unit_test"
        }
        self.dummpy_model_properties = [
            {
                # from url
                "name": "ViT-B-32",
                "dimensions" : 512,
                "url": "https://a-url-to-a-corrupted-model.pt",
                "type": "open_clip",
            },
            {
                # from s3
                "name": "ViT-B-32",
                "dimensions" : 512,
                "model_location": {
                    "s3":{
                        "Bucket": "a-bucket",
                        "Key": "a-path-to-a-corrupted-model.pt"},
                        },
                "type": "open_clip",
            },
            {
                # from hf
                "name": "ViT-B-32",
                "dimensions": 512,
                "model_location": {
                    "hf": {
                        "repo_id" : "a-dummy-repo",
                        "filename": "a-path-to-a-corrupted-model.pt"},
                        },
                "type": "open_clip",
            }
        ]

        self.dummpy_corrupted_file = "/path/to/corrupted/file.pt"

    @patch('open_clip.create_model_and_transforms', autospec=True)
    @patch('os.remove', autospec=True)
    def test_corrupted_file_handling(self, mock_os_remove, mock_create_model_and_transforms):
        # Setup
        mock_create_model_and_transforms.side_effect = RuntimeError("The file might be corrupted")
        with patch("marqo.s2_inference.clip_utils.download_model", return_value = self.dummpy_corrupted_file):
            for model_properties in self.dummpy_model_properties:
                with self.assertRaises(InvalidModelPropertiesError) as context:
                    _ = _load_model(**self.load_parameters, model_properties=model_properties)
                # Verify
                self.assertIn("Marqo encountered a corrupted file when loading open_clip file", str(context.exception))
                mock_os_remove.assert_called_once_with(self.dummpy_corrupted_file)

                # Reset necessary mock
                mock_os_remove.reset_mock()

    @patch('open_clip.create_model_and_transforms', autospec=True)
    @patch('os.remove', autospec=True)
    def test_file_removal_failure_handling(self, mock_os_remove, mock_create_model_and_transforms):
        # Setup
        mock_create_model_and_transforms.side_effect = RuntimeError("The file might be corrupted")
        mock_os_remove.side_effect = OSError("Permission denied")
        with patch("marqo.s2_inference.clip_utils.download_model", return_value = self.dummpy_corrupted_file):
            for model_properties in self.dummpy_model_properties:
                # Execute and Verify
                with self.assertRaises(InvalidModelPropertiesError) as context:
                    _ = _load_model(**self.load_parameters, model_properties=model_properties)
                self.assertIn("Marqo encountered an error while attempting to delete corrupted file",
                              str(context.exception))
                mock_os_remove.assert_called_once_with(self.dummpy_corrupted_file)


                # Reset the mock
                mock_os_remove.reset_mock()

    @patch('open_clip.create_model_and_transforms', autospec=True)
    @patch('os.remove', autospec=True)
    def test_other_errors_handling(self, mock_os_remove, mock_create_model_and_transforms):
        # Setup
        mock_create_model_and_transforms.side_effect = Exception("An error occurred")
        with patch("marqo.s2_inference.clip_utils.download_model", return_value = self.dummpy_corrupted_file):
            for model_properties in self.dummpy_model_properties:
                # Execute and Verify
                with self.assertRaises(InvalidModelPropertiesError) as context:
                    _ = _load_model(**self.load_parameters, model_properties=model_properties)
                self.assertIn("Marqo encountered an error when loading custom open_clip model", str(context.exception))
                mock_os_remove.assert_not_called()

    @patch('open_clip.create_model_and_transforms', autospec=True)
    @patch('os.remove', autospec=True)
    def test_load_clip_into_open_clip_errors_handling(self, mock_os_remove, mock_create_model_and_transforms):
        # Setup
        mock_create_model_and_transforms.side_effect = Exception(
            "This could be because the operator doesn't exist for this backend")
        with patch("marqo.s2_inference.clip_utils.download_model", return_value = self.dummpy_corrupted_file):
            for model_properties in self.dummpy_model_properties:
                # Execute and Verify
                with self.assertRaises(InvalidModelPropertiesError) as context:
                    _ = _load_model(**self.load_parameters, model_properties=model_properties)
                self.assertIn(
                    "It is likely that you are tyring to load a `CLIP (OpenAI)` model with type `open_clip` in model_properties.",
                    str(context.exception))
                mock_os_remove.assert_not_called()

    def test_load_clip_model_into_open_clip_no_mock(self):
        model_properties = {
            "name": "ViT-B-32",
            "dimensions": 512,
            "url": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
            "type": "open_clip",
        }
        with self.assertRaises(InvalidModelPropertiesError) as context:
            _ = _load_model(**self.load_parameters, model_properties=model_properties)

        self.assertIn(
            "It is likely that you are tyring to load a `CLIP (OpenAI)` model with type `open_clip` in model_properties.",
            str(context.exception))


class TestCorruptFileInHuggingFace(unittest.TestCase):
    '''Test that a corrupt file in HuggingFace is handled correctly.
        HuggingFace normally load a directory. So the checking step is normally done in the extraction step.

        We DO NOT test download integrity for the HF model loaded from Hugging Face model card. This is managed
        by Hugging Face directly.
    '''
    def setUp(self):
        self.load_parameters = {
            "model_name": "test-corrupted-open-clip-model",
            "device": "cpu",
            "model_auth": None,
            "calling_func": "unit_test"
        }
        self.dummy_model_properties = [
            {
                # from url
                "dimensions": 384,
                "url": "https://a-url-to-a-corrupted-model.zip",
                "type": "hf",
            },
            {
                # from s3
                "dimensions": 384,
                "model_location": {
                    "s3": {
                        "Bucket": "a-bucket",
                        "Key": "a-path-to-a-corrupted-model.zip"},
                },
                "type": "hf",
            },
            {
                # from hf
                "dimensions": 384,
                "model_location": {
                    "hf": {
                        "repo_id": "a-dummy-repo",
                        "filename": "a-path-to-a-corrupted-model.zip"},
                },
                "type": "hf",
            }
        ]

    def test_regular_file(self):
        with patch('os.path.isfile', return_value=True), \
             patch('os.path.splitext', return_value=('/path/to/file', '.txt')), \
             patch('os.makedirs'), \
             patch("marqo.s2_inference.hf_utils.download_model", return_value = "/path/to/file.txt"):
            for model_properties in self.dummy_model_properties:
                with self.assertRaises(InvalidModelPropertiesError) as context:
                    _ = _load_model(**self.load_parameters, model_properties=model_properties)
                self.assertIn("No such file or directory: '/path/to/file.txt", str(context.exception))

    def test_zip_file(self):
        with patch('os.path.isfile', return_value=True), \
             patch('os.path.splitext', return_value=('/path/to/file', '.zip')), \
             patch('os.makedirs') as mock_makedirs, \
             patch('zipfile.ZipFile') as mock_zipfile, \
             patch("marqo.s2_inference.hf_utils.download_model", return_value="/path/to/file.zip"),\
             patch("transformers.AutoModel.from_pretrained") as mock_model,\
             patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer:

            for model_properties in self.dummy_model_properties:
                _ = _load_model(**self.load_parameters, model_properties=model_properties)

                mock_makedirs.assert_called_once_with('/path/to/file', exist_ok=True)
                mock_zipfile.assert_called_once_with('/path/to/file.zip', 'r')
                mock_model.assert_called_once_with('/path/to/file', use_auth_token=None, cache_dir = ModelCache.hf_cache_path)
                mock_tokenizer.assert_called_once_with('/path/to/file')

                mock_makedirs.reset_mock()
                mock_zipfile.reset_mock()
                mock_model.reset_mock()
                mock_tokenizer.reset_mock()

    def test_tar_file(self):
        with patch('os.path.isfile', return_value=True), \
            patch('os.path.splitext', return_value=('/path/to/file', '.tar')), \
            patch('os.makedirs') as mock_makedirs, \
            patch('tarfile.open') as mock_tarfile,\
            patch("marqo.s2_inference.hf_utils.download_model", return_value="/path/to/file.tar"), \
            patch("transformers.AutoModel.from_pretrained") as mock_model, \
            patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer:

            for model_properties in self.dummy_model_properties:
                _ = _load_model(**self.load_parameters, model_properties=model_properties)

                mock_makedirs.assert_called_once_with('/path/to/file', exist_ok=True)
                mock_tarfile.assert_called_once_with('/path/to/file.tar', 'r')
                mock_model.assert_called_once_with('/path/to/file', use_auth_token=None, cache_dir=ModelCache.hf_cache_path)
                mock_tokenizer.assert_called_once_with('/path/to/file')

                mock_makedirs.reset_mock()
                mock_tarfile.reset_mock()
                mock_model.reset_mock()
                mock_tokenizer.reset_mock()

    def test_directory(self):
        with patch('os.path.isfile', return_value=False),\
            patch("marqo.s2_inference.hf_utils.download_model", return_value="/path/to/file.tar"), \
            patch("transformers.AutoModel.from_pretrained") as mock_model, \
            patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer:
            self.assertEqual(extract_huggingface_archive('/path/to/directory'), '/path/to/directory')

    def test_hf_repo_id(self):
        with patch('os.path.isfile', return_value=False):
            self.assertEqual(extract_huggingface_archive('sentence-transformers/all-MiniLM-L6-v2'), 'sentence-transformers/all-MiniLM-L6-v2')

    def test_extraction_failure(self):
        with patch('os.path.isfile', return_value=True), \
             patch('os.path.splitext', return_value=('/path/to/file', '.zip')), \
             patch('os.makedirs'), \
             patch('zipfile.ZipFile', side_effect=zipfile.BadZipfile), \
             patch('os.remove') as mock_remove:
            with self.assertRaises(InvalidModelPropertiesError):
                extract_huggingface_archive('/path/to/file.zip')
            mock_remove.assert_called_once_with('/path/to/file.zip')
