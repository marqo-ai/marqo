import uuid

from marqo.client import Client

from tests.marqo_test import MarqoTestCase
from marqo.errors import MarqoWebError


class TestAddDocumentsCommon(MarqoTestCase):
    """A class to test common add_documents functionalities for structured and unstructured indexes.

    We should test the shared functionalities between structured and unstructured indexes here to avoid code duplication
    and branching in the test cases."""

    structured_text_index_name = "structured_index_text" + str(uuid.uuid4()).replace('-', '')
    structured_image_index_name = "structured_image_index" + str(uuid.uuid4()).replace('-', '')
    structured_filter_index_name = "structured_filter_index" + str(uuid.uuid4()).replace('-', '')

    unstructured_text_index_name = "unstructured_index_text" + str(uuid.uuid4()).replace('-', '')
    unstructured_image_index_name = "unstructured_image_index" + str(uuid.uuid4()).replace('-', '')

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = Client(**cls.client_settings)

        cls.create_indexes([
            {
                "indexName": cls.structured_text_index_name,
                "type": "structured",
                "model": "hf/all-MiniLM-L6-v2",
                "allFields": [
                    {"name": "title", "type": "text", "features": ["filter", "lexical_search"]},
                    {"name": "content", "type": "text", "features": ["filter", "lexical_search"]},
                ],
                "tensorFields": ["title", "content"],
            },
            {
                "indexName": cls.structured_filter_index_name,
                "type": "structured",
                "model": "hf/all-MiniLM-L6-v2",
                "allFields": [
                    {"name": "field_a", "type": "text", "features": ["filter", "lexical_search"]},
                    {"name": "field_b", "type": "text", "features": ["filter"]},
                    {"name": "str_for_filtering", "type": "text", "features": ["filter"]},
                    {"name": "int_for_filtering", "type": "int", "features": ["filter"]},
                    {"name": "long_field_1", "type": "long", "features": ["filter"]},
                    {"name": "double_field_1", "type": "double", "features": ["filter"]},
                    {"name": "array_long_field_1", "type": "array<long>", "features": ["filter"]},
                    {"name": "array_double_field_1", "type": "array<double>", "features": ["filter"]}
                ],
                "tensorFields": ["field_a", "field_b"],
            },
            {
                "indexName": cls.structured_image_index_name,
                "type": "structured",
                "model": "open_clip/ViT-B-32/openai",
                "allFields": [
                    {"name": "title", "type": "text", "features": ["filter", "lexical_search"]},
                    {"name": "content", "type": "text", "features": ["filter", "lexical_search"]},
                    {"name": "image_content", "type": "image_pointer"},
                    {"name": "image_field_1", "type": "image_pointer"},
                    {"name": "text_field_1", "type": "text", "features": ["filter", "lexical_search"]},
                ],
                "tensorFields": ["title", "image_content", "image_field_1"],
            }
        ])

        cls.create_indexes([
            {
                "indexName": cls.unstructured_text_index_name,
                "type": "unstructured",
                "model": "hf/all-MiniLM-L6-v2",
            },
            {
                "indexName": cls.unstructured_image_index_name,
                "type": "unstructured",
                "model": "open_clip/ViT-B-32/openai",
                "treatUrlsAndPointersAsMedia": True
            }
        ])

        cls.indexes_to_delete = [cls.structured_image_index_name, cls.structured_filter_index_name,
                                 cls.structured_text_index_name, cls.unstructured_image_index_name,
                                 cls.unstructured_text_index_name]

    def test_add_documents_for_private_images(self):
        """A test to add documents with private images using media_download_headers and image_download_headers."""
        documents = [
            {
                "image_field_1": "https://d2k91vq0avo7lq.cloudfront.net/ai_hippo_realistic_small.png",
                "text_field_1": "A private image with a png extension",
                "_id": "1"
            },
            {
                "image_field_1": "https://d2k91vq0avo7lq.cloudfront.net/ai_hippo_realistic_small",
                "text_field_1": "A private image without an extension",
                "_id": "2"
            }
        ]

        kwargs_list = [
            {"media_download_headers": {"marqo_media_header": "media_header_test_key"}},
            {"image_download_headers": {"marqo_media_header": "media_header_test_key"}}
        ]

        for index_name in [self.unstructured_image_index_name, self.structured_image_index_name]:
            tensor_fields = ["image_field_1"] if (
                    index_name == self.unstructured_image_index_name) else None
            for kwargs in kwargs_list:
                with self.subTest(f"{index_name} - {kwargs}"):
                    res = self.client.index(index_name).add_documents(
                        documents, tensor_fields=tensor_fields,
                        **kwargs
                    )
                    self.assertEqual(False, res["errors"], res)
                    self.assertEqual(2, self.client.index(index_name).get_stats()["numberOfDocuments"])
                    self.assertEqual(2, self.client.index(index_name).get_stats()["numberOfVectors"])

    def test_proper_error_when_adding_documents_with_private_image_without_access(self):
        """A test to check that the proper error is raised when adding documents with private images without access."""
        documents = [
            {
                "image_field_1": "https://d2k91vq0avo7lq.cloudfront.net/ai_hippo_realistic_small.png",
                "text_field_1": "A private image with a png extension",
                "_id": "1"
            },
            {
                "image_field_1": "https://d2k91vq0avo7lq.cloudfront.net/ai_hippo_realistic_small",
                "text_field_1": "A private image without an extension",
                "_id": "2"
            }
        ]
        for index_name in [self.unstructured_image_index_name, self.structured_image_index_name]:
            tensor_fields = ["image_field_1"] if (
                    index_name == self.unstructured_image_index_name
            ) else None
            with self.subTest(f"{index_name}"):
                res = self.client.index(index_name).add_documents(
                    documents, tensor_fields=tensor_fields,
                )
                self.assertEqual(True, res["errors"], res)
                for item in res["items"]:
                    self.assertEqual(400, item["status"], item)
                    self.assertIn("403", item["error"], item)

    def test_proper_error_if_both_imageDownloadHeaders_and_mediaDownloadHeaders_are_provided(self):
        """Test that an error is raised if both imageDownloadHeaders and mediaDownloadHeaders are provided."""
        documents = [
            {
                "image_field_1": "https://d2k91vq0avo7lq.cloudfront.net/ai_hippo_realistic_small.png",
                "text_field_1": "A private image with a png extension",
                "_id": "1"
            },
            {
                "image_field_1": "https://d2k91vq0avo7lq.cloudfront.net/ai_hippo_realistic_small",
                "text_field_1": "A private image without an extension",
                "_id": "2"
            }
        ]
        for index_name in [self.unstructured_image_index_name, self.structured_image_index_name]:
            tensor_fields = ["image_field_1"] if (
                    index_name == self.unstructured_image_index_name) else None
            with self.assertRaises(MarqoWebError) as cm:
                res = self.client.index(index_name).add_documents(
                    documents, tensor_fields=tensor_fields,
                    image_download_headers={"marqo_media_header": "media_header_test_key"},
                    media_download_headers={"marqo_media_header": "media_header_test_key"}
                )
                self.assertIn("Cannot set both imageDownloadHeaders and mediaDownloadHeaders.",
                              str(cm.exception.message))
