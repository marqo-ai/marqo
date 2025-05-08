import uuid

from marqo.errors import MarqoWebError

from tests.marqo_test import MarqoTestCase


def generate_structured_index_settings_dict(index_name, image_preprocessing_method):
    return {
        "indexName": index_name,
        "type": "structured",
        "model": "open_clip/ViT-B-32/openai",
        "allFields": [{"name": "image_content_1", "type": "image_pointer"},
                      {"name": "image_content_2", "type": "image_pointer"},
                      {"name": "text_content", "type": "text"}],
        "tensorFields": ["image_content_1", "image_content_2", "text_content"],
        "imagePreprocessing": {"patchMethod": image_preprocessing_method}
    }


class TestImageReranking(MarqoTestCase):
    """Test image reranking features. Note that this feature is available only for structured indexes as
    the feature requires searchable attributes."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.structured_no_image_processing_index_name = (
                "structured_no_image_processing_index_name" + str(uuid.uuid4()).replace('-', ''))

        cls.create_indexes([
            generate_structured_index_settings_dict(cls.structured_no_image_processing_index_name, None),
        ])

        cls.indexes_to_delete = [
            cls.structured_no_image_processing_index_name,
        ]

    def test_reranking_not_supported(self):
        with self.assertRaises(MarqoWebError) as e:
            _ = self.client.index(self.structured_no_image_processing_index_name).search(
                'brain', reranker='google/owlvit-base-patch32')
        self.assertIn('Reranker is no longer supported in Marqo version 2.17 and later', str(e.exception.message))
