import unittest
from pickle import UnpicklingError
from unittest.mock import patch

from marqo.s2_inference.errors import InvalidModelPropertiesError
from marqo.s2_inference.s2_inference import _load_model


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
        self.dummy_model_properties = [
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

    @patch('open_clip.create_model', autospec=True)
    @patch('os.remove', autospec=True)
    def test_corrupted_file_handling_runtime_error(self, mock_os_remove, mock_create_model_and_transforms):
        """Ensure that a proper error is raised when a corrupted file is encountered. The file should be removed."""
        mock_create_model_and_transforms.side_effect = RuntimeError("The file might be corrupted")
        for model_properties in self.dummy_model_properties:
            with patch("marqo.inference.native_inference.embedding_models.open_clip_model.download_model",
                       return_value = self.dummpy_corrupted_file):
                with self.assertRaises(InvalidModelPropertiesError) as context:
                    _ = _load_model(**self.load_parameters, model_properties=model_properties, max_retries=1)
                # Verify
                self.assertIn("Marqo encountered a corrupted file when loading open_clip file", str(context.exception))
                mock_os_remove.assert_called_once_with(self.dummpy_corrupted_file)

                # Reset the mock
                mock_os_remove.reset_mock()

    @patch('open_clip.create_model', autospec=True)
    @patch('os.remove', autospec=True)
    def test_corrupted_file_handling_unpickling_error(self, mock_os_remove, mock_create_model_and_transforms):
        """Ensure that a proper error is raised when a corrupted file is encountered. The file should be removed."""
        mock_create_model_and_transforms.side_effect = UnpicklingError("The file might be corrupted")
        for model_properties in self.dummy_model_properties:
            with patch("marqo.inference.native_inference.embedding_models.open_clip_model.download_model",
                       return_value=self.dummpy_corrupted_file):
                with self.assertRaises(InvalidModelPropertiesError) as context:
                    _ = _load_model(**self.load_parameters, model_properties=model_properties, max_retries=1)
                # Verify
                self.assertIn("Marqo encountered a corrupted file when loading open_clip file", str(context.exception))
                mock_os_remove.assert_called_once_with(self.dummpy_corrupted_file)

                # Reset the mock
                mock_os_remove.reset_mock()

    @patch('open_clip.create_model', autospec=True)
    @patch('os.remove', autospec=True)
    def test_file_removal_failure_handling(self, mock_os_remove, mock_create_model_and_transforms):
        # Setup
        mock_create_model_and_transforms.side_effect = RuntimeError("The file might be corrupted")
        mock_os_remove.side_effect = OSError("Permission denied")
        with patch("marqo.inference.native_inference.embedding_models.open_clip_model.download_model",
                   return_value = self.dummpy_corrupted_file):
            for model_properties in self.dummy_model_properties:
                # Execute and Verify
                with self.assertRaises(RuntimeError) as context:
                    _ = _load_model(**self.load_parameters, model_properties=model_properties, max_retries=1)
                self.assertIn("Marqo encountered an error while attempting to delete a corrupted file",
                              str(context.exception))
                mock_os_remove.assert_called_with(self.dummpy_corrupted_file)
                self.assertEqual(mock_os_remove.call_count, 1)

                # Reset the mock
                mock_os_remove.reset_mock()

    @patch('open_clip.create_model', autospec=True)
    @patch('os.remove', autospec=True)
    def test_other_errors_handling(self, mock_os_remove, mock_create_model_and_transforms):
        # Setup
        mock_create_model_and_transforms.side_effect = Exception("An error occurred")
        with patch("marqo.inference.native_inference.embedding_models.open_clip_model.download_model",
                   return_value = self.dummpy_corrupted_file):
            for model_properties in self.dummy_model_properties:
                # Execute and Verify
                with self.assertRaises(RuntimeError) as context:
                    _ = _load_model(**self.load_parameters, model_properties=model_properties, max_retries=1)
                self.assertIn("Marqo encountered an error when loading custom open_clip model", str(context.exception))
                mock_os_remove.assert_not_called()

    @patch('open_clip.create_model', autospec=True)
    @patch('os.remove', autospec=True)
    def test_load_clip_into_open_clip_errors_handling(self, mock_os_remove, mock_create_model_and_transforms):
        # Setup
        mock_create_model_and_transforms.side_effect = Exception(
            "This could be because the operator doesn't exist for this backend")
        with patch("marqo.inference.native_inference.embedding_models.open_clip_model.download_model",
                   return_value=self.dummpy_corrupted_file):
            for model_properties in self.dummy_model_properties:
                # Execute and Verify
                with self.assertRaises(InvalidModelPropertiesError) as context:
                    _ = _load_model(**self.load_parameters, model_properties=model_properties, max_retries=1)
                self.assertIn(
                    "You may have tried to load a clip model even though model_properties['type'] is set to 'open_clip'",
                    str(context.exception))
                mock_os_remove.assert_not_called()