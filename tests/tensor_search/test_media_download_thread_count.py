import unittest
from unittest.mock import patch, MagicMock
from marqo.tensor_search.add_docs import _determine_thread_count
from marqo.tensor_search.enums import EnvVars
import os

# Mock classes
class MarqoIndex:
    def __init__(self, model_type):
        self.model = MagicMock()
        self.model.get_properties.return_value = {'type': model_type}

class AddDocsParams:
    def __init__(self, media_download_thread_count=None, image_download_thread_count=None):
        self.media_download_thread_count = media_download_thread_count
        self.image_download_thread_count = image_download_thread_count

class TestDetermineThreadCount(unittest.TestCase):

    def setUp(self):
        # Clear environment variables before each test
        self.env_patcher = patch.dict(os.environ, {}, clear=True)
        self.env_patcher.start()

    def tearDown(self):
        # Stop patching environment variables
        self.env_patcher.stop()

    def test_defaults_with_non_languagebind_model(self):
        """Test that default image thread count is returned when model is not languagebind and no thread counts are set."""
        marqo_index = MarqoIndex(model_type='other')
        add_docs_params = AddDocsParams()
        result = _determine_thread_count(marqo_index, add_docs_params)
        self.assertEqual(result, 20)

    def test_defaults_with_languagebind_model(self):
        """Test that default media thread count is returned when model is languagebind and no thread counts are set."""
        marqo_index = MarqoIndex(model_type='languagebind')
        add_docs_params = AddDocsParams()
        result = _determine_thread_count(marqo_index, add_docs_params)
        self.assertEqual(result, 5)

    def test_media_thread_count_set_in_params(self):
        """Test that media_download_thread_count from params overrides defaults."""
        marqo_index = MarqoIndex(model_type='other')
        add_docs_params = AddDocsParams(media_download_thread_count=10)
        result = _determine_thread_count(marqo_index, add_docs_params)
        self.assertEqual(result, 10)

    def test_media_thread_count_set_in_env(self):
        """Test that media_download_thread_count from environment variable overrides defaults."""
        marqo_index = MarqoIndex(model_type='other')
        add_docs_params = AddDocsParams()
        os.environ[EnvVars.MARQO_MEDIA_DOWNLOAD_THREAD_COUNT_PER_REQUEST] = '15'
        result = _determine_thread_count(marqo_index, add_docs_params)
        self.assertEqual(result, 15)

    def test_image_thread_count_set_in_params(self):
        """Test that image_download_thread_count from params overrides defaults when media thread counts are not set."""
        marqo_index = MarqoIndex(model_type='other')
        add_docs_params = AddDocsParams(image_download_thread_count=25)
        result = _determine_thread_count(marqo_index, add_docs_params)
        self.assertEqual(result, 25)

    def test_image_thread_count_set_in_env(self):
        """Test that image_download_thread_count from environment variable overrides defaults when media thread counts are not set."""
        marqo_index = MarqoIndex(model_type='other')
        add_docs_params = AddDocsParams()
        os.environ[EnvVars.MARQO_IMAGE_DOWNLOAD_THREAD_COUNT_PER_REQUEST] = '30'
        result = _determine_thread_count(marqo_index, add_docs_params)
        self.assertEqual(result, 30)

    def test_media_thread_count_defaults_to_image_thread_count(self):
        """Test that when media thread counts are not set and model is not languagebind, defaults to image thread count."""
        marqo_index = MarqoIndex(model_type='other')
        add_docs_params = AddDocsParams()
        result = _determine_thread_count(marqo_index, add_docs_params)
        self.assertEqual(result, 20)

    def test_media_thread_count_equals_default(self):
        """Test that when media_download_thread_count equals default, it is ignored and defaults are applied."""
        marqo_index = MarqoIndex(model_type='other')
        add_docs_params = AddDocsParams(media_download_thread_count=5)
        result = _determine_thread_count(marqo_index, add_docs_params)
        self.assertEqual(result, 20)

    def test_media_env_thread_count_equals_default(self):
        """Test that when media thread count in env equals default, it is ignored and defaults are applied."""
        marqo_index = MarqoIndex(model_type='other')
        add_docs_params = AddDocsParams()
        os.environ[EnvVars.MARQO_MEDIA_DOWNLOAD_THREAD_COUNT_PER_REQUEST] = '5'
        result = _determine_thread_count(marqo_index, add_docs_params)
        self.assertEqual(result, 20)

    def test_image_thread_count_equals_default(self):
        """Test that when image_download_thread_count equals default, it is ignored and defaults are applied."""
        marqo_index = MarqoIndex(model_type='other')
        add_docs_params = AddDocsParams(image_download_thread_count=20)
        result = _determine_thread_count(marqo_index, add_docs_params)
        self.assertEqual(result, 20)

    def test_image_env_thread_count_equals_default(self):
        """Test that when image thread count in env equals default, it is ignored and defaults are applied."""
        marqo_index = MarqoIndex(model_type='other')
        add_docs_params = AddDocsParams()
        os.environ[EnvVars.MARQO_IMAGE_DOWNLOAD_THREAD_COUNT_PER_REQUEST] = '20'
        result = _determine_thread_count(marqo_index, add_docs_params)
        self.assertEqual(result, 20)

    def test_both_thread_counts_set_in_params_non_languagebind_model(self):
        """Test when both media_download_thread_count and image_download_thread_count are set in params for non-languagebind model."""
        marqo_index = MarqoIndex(model_type='other')
        add_docs_params = AddDocsParams(media_download_thread_count=8, image_download_thread_count=15)
        result = _determine_thread_count(marqo_index, add_docs_params)
        self.assertEqual(result, 8)

    def test_media_thread_count_in_env_image_thread_count_in_params_non_languagebind_model(self):
        """Test when media thread count is set in env and image thread count in params for non-languagebind model."""
        marqo_index = MarqoIndex(model_type='other')
        add_docs_params = AddDocsParams(image_download_thread_count=15)
        os.environ[EnvVars.MARQO_MEDIA_DOWNLOAD_THREAD_COUNT_PER_REQUEST] = '8'
        result = _determine_thread_count(marqo_index, add_docs_params)
        self.assertEqual(result, 8)

    def test_media_thread_count_in_params_image_thread_count_in_env_non_languagebind_model(self):
        """Test when media thread count is set in params and image thread count in env for non-languagebind model."""
        marqo_index = MarqoIndex(model_type='other')
        add_docs_params = AddDocsParams(media_download_thread_count=8)
        os.environ[EnvVars.MARQO_IMAGE_DOWNLOAD_THREAD_COUNT_PER_REQUEST] = '15'
        result = _determine_thread_count(marqo_index, add_docs_params)
        self.assertEqual(result, 8)

    def test_both_thread_counts_set_in_env_non_languagebind_model(self):
        """Test when both media_download_thread_count and image_download_thread_count are set in env for non-languagebind model."""
        marqo_index = MarqoIndex(model_type='other')
        add_docs_params = AddDocsParams()
        os.environ[EnvVars.MARQO_MEDIA_DOWNLOAD_THREAD_COUNT_PER_REQUEST] = '8'
        os.environ[EnvVars.MARQO_IMAGE_DOWNLOAD_THREAD_COUNT_PER_REQUEST] = '15'
        result = _determine_thread_count(marqo_index, add_docs_params)
        self.assertEqual(result, 8)

    def test_media_thread_count_equals_default_image_thread_count_set_non_languagebind_model(self):
        """Test when media_download_thread_count equals default and image_download_thread_count is set for non-languagebind model."""
        marqo_index = MarqoIndex(model_type='other')
        add_docs_params = AddDocsParams(media_download_thread_count=5, image_download_thread_count=15)
        result = _determine_thread_count(marqo_index, add_docs_params)
        self.assertEqual(result, 15)

    def test_media_thread_count_set_languagebind_model(self):
        """Test that media_download_thread_count from params is used for languagebind model."""
        marqo_index = MarqoIndex(model_type='languagebind')
        add_docs_params = AddDocsParams(media_download_thread_count=8)
        result = _determine_thread_count(marqo_index, add_docs_params)
        self.assertEqual(result, 8)

    def test_image_thread_count_set_languagebind_model(self):
        """Test that image_download_thread_count is ignored for languagebind model when media_download_thread_count is not set."""
        marqo_index = MarqoIndex(model_type='languagebind')
        add_docs_params = AddDocsParams(image_download_thread_count=15)
        result = _determine_thread_count(marqo_index, add_docs_params)
        self.assertEqual(result, 5)  # Should default to 5 for languagebind model

    def test_media_thread_count_equals_default_languagebind_model(self):
        """Test when media_download_thread_count equals default for languagebind model."""
        marqo_index = MarqoIndex(model_type='languagebind')
        add_docs_params = AddDocsParams(media_download_thread_count=5)
        result = _determine_thread_count(marqo_index, add_docs_params)
        self.assertEqual(result, 5)

    def test_media_thread_count_not_set_image_thread_count_set_languagebind_model(self):
        """Test when media_download_thread_count is not set and image_download_thread_count is set for languagebind model."""
        marqo_index = MarqoIndex(model_type='languagebind')
        add_docs_params = AddDocsParams(image_download_thread_count=15)
        result = _determine_thread_count(marqo_index, add_docs_params)
        self.assertEqual(result, 5)  # Should ignore image thread count for languagebind