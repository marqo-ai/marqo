import uuid

from marqo.client import Client
from marqo.errors import MarqoWebError

from tests.marqo_test import MarqoTestCase


class TestSearchCommon(MarqoTestCase):
    """A class to test common search functionalities for structured and unstructured indexes.

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
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "allFields": [
                    {"name": "title", "type": "text", "features": ["filter", "lexical_search"]},
                    {"name": "content", "type": "text", "features": ["filter", "lexical_search"]},
                ],
                "tensorFields": ["title", "content"],
            },
            {
                "indexName": cls.structured_filter_index_name,
                "type": "structured",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
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
                    {"name": "text_field_1", "type": "text", "features": ["filter", "lexical_search"]},
                    {"name": "image_content", "type": "image_pointer"},
                    {"name": "image_field_1", "type": "image_pointer"},
                ],
                "tensorFields": ["title", "image_content", "image_field_1"],
            }
        ])

        cls.create_indexes([
            {
                "indexName": cls.unstructured_text_index_name,
                "type": "unstructured",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
            },
            {
                "indexName": cls.unstructured_image_index_name,
                "type": "unstructured",
                "model": "open_clip/ViT-B-32/openai"
            }
        ])

        cls.indexes_to_delete = [cls.structured_image_index_name, cls.structured_filter_index_name,
                                 cls.structured_text_index_name, cls.unstructured_image_index_name,
                                 cls.unstructured_text_index_name]

    def test_lexical_query_can_not_be_none(self):
        context = {"tensor": [{"vector": [1, ] * 384, "weight": 1},
                          {"vector": [2, ] * 384, "weight": 2}]}

        test_case = [
            (None, context, "with context"),
            (None, None, "without context")
        ]
        for index_name in [self.structured_text_index_name, self.unstructured_image_index_name]:
            for query, context, msg in test_case:
                with self.subTest(f"{index_name} - {msg}"):
                    with self.assertRaises(MarqoWebError) as e:
                        res = self.client.index(index_name).search(q=None, context=context, search_method="LEXICAL")
                    self.assertIn("Query(q) is required for lexical search", str(e.exception.message))

    def test_tensor_search_query_can_be_none(self):
        context = {"tensor": [{"vector": [1, ] * 384, "weight": 1},
                          {"vector": [2, ] * 384, "weight": 2}]}
        for index_name in [self.structured_text_index_name, self.unstructured_text_index_name]:
            res = self.client.index(index_name).search(q=None, context=context)
            self.assertIn("hits", res)

    def test_add_document_and_search_for_private_images(self):
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
            res = self.client.index(index_name).add_documents(
                documents, tensor_fields=tensor_fields,
                media_download_headers={"marqo_media_header": "media_header_test_key"}
            )

            for kwargs in kwargs_list:
                for query in [
                    "https://d2k91vq0avo7lq.cloudfront.net/ai_hippo_realistic_small",
                    {
                        "https://d2k91vq0avo7lq.cloudfront.net/ai_hippo_realistic_small.png": 1,
                        "A private image without an extension": 1
                    }
                ]:

                    with self.subTest(f"{index_name} - {kwargs} - {query}"):
                        res = self.client.index(index_name).search(query, **kwargs)
                        self.assertIn("hits", res, res)
                        self.assertEqual(2, len(res["hits"]), res)

    def test_invalidArgError_is_raised_when_searching_a_private_image(self):
        query= "https://d2k91vq0avo7lq.cloudfront.net/ai_hippo_realistic_small"
        for index_name in [self.structured_image_index_name, self.unstructured_image_index_name]:
            with self.subTest(f"{index_name}"):
                with self.assertRaises(MarqoWebError) as e:
                    self.client.index(index_name).search(query)
                self.assertIn("Error downloading media file", str(e.exception))

    def test_proper_error_if_both_imageDownloadHeaders_and_mediaDownloadHeaders_are_provided(self):
        """Test that an error is raised if both imageDownloadHeaders and mediaDownloadHeaders are provided."""
        for index_name in [self.unstructured_image_index_name, self.structured_image_index_name]:
            with self.assertRaises(MarqoWebError) as cm:
                res = self.client.index(index_name).search(
                    "test",
                    image_download_headers={"marqo_media_header": "media_header_test_key"},
                    media_download_headers={"marqo_media_header": "media_header_test_key"}
                )
                self.assertIn("Cannot set both imageDownloadHeaders and mediaDownloadHeaders.",
                              str(cm.exception.message))