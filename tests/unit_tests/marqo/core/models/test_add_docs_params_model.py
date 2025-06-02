from unittest import TestCase
from unittest.mock import patch

from pydantic.v1 import ValidationError

from marqo.core.models.add_docs_params import AddDocsParams
from marqo.tensor_search.enums import EnvVars


class TestAddDocsParamsModel(TestCase):

    def setUp(self):
        self.sample_docs = [{"doc1": "value1"}, {"doc2": "value2"}]

    def test_both_thread_counts_none(self):
        """Both counts None -> should read defaults."""
        params = AddDocsParams(
            docs=self.sample_docs,
            index_name="test_index",
            image_download_thread_count=None,
            media_download_thread_count=None,
            device=None
        )

        self.assertEqual(20, params.image_download_thread_count)
        self.assertEqual(5, params.media_download_thread_count)

    def test_only_image_thread_count_given(self):
        """Image count set, media None -> media gets default, image uses given."""
        params = AddDocsParams(
            docs=self.sample_docs,
            index_name="test_index",
            image_download_thread_count=3,
            media_download_thread_count=None,
            device=None
        )

        self.assertEqual(3, params.image_download_thread_count)
        self.assertEqual(5, params.media_download_thread_count)

    def test_only_media_thread_count_given(self):
        """Media count set, image None -> image equals media."""
        params = AddDocsParams(
            docs=self.sample_docs,
            index_name="test_index",
            image_download_thread_count=None,
            media_download_thread_count=4,
            device=None
        )

        self.assertEqual(4, params.media_download_thread_count)
        self.assertEqual(4, params.image_download_thread_count)

    def test_both_thread_counts_set_should_fail(self):
        """Both set -> should raise ValueError."""
        with self.assertRaises(ValidationError) as context:
            AddDocsParams(
                docs=self.sample_docs,
                index_name="test_index",
                image_download_thread_count=2,
                media_download_thread_count=3,
                device=None
            )
        self.assertIn("Cannot set both", str(context.exception))

    @patch("marqo.core.models.add_docs_params.read_env_vars_and_defaults_ints")
    def test_env_vars_override_defaults(self, mock_read_env_vars):

        def mock_env_var_reader(key):
            if key == EnvVars.MARQO_IMAGE_DOWNLOAD_THREAD_COUNT_PER_REQUEST:
                return 15
            elif key == EnvVars.MARQO_MEDIA_DOWNLOAD_THREAD_COUNT_PER_REQUEST:
                return 7
            elif key == EnvVars.MARQO_MAX_DOCUMENTS_BATCH_SIZE:
                return 100
            return 1  # fallback

        mock_read_env_vars.side_effect = mock_env_var_reader

        params = AddDocsParams(
            docs=self.sample_docs,
            index_name="test_index",
            image_download_thread_count=None,
            media_download_thread_count=None,
            device=None
        )

        # Validate that environment values were respected
        self.assertEqual(params.image_download_thread_count, 15)
        self.assertEqual(params.media_download_thread_count, 7)