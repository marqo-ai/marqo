import uuid
from marqo.errors import MarqoWebError

import numpy as np

from tests.marqo_test import MarqoTestCase



class TestEmbed(MarqoTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.structured_index_name = "structured_" + str(uuid.uuid4()).replace('-', '')
        cls.unstructured_index_name = "unstructured_" + str(uuid.uuid4()).replace('-', '')
        cls.unstructured_index_non_e5 = "unstructured_non_e5_" + str(uuid.uuid4()).replace('-', '')

        cls.structured_image_index_name = "structured_image_index" + str(uuid.uuid4()).replace('-', '')
        cls.unstructured_image_index_name = "unstructured_image_index" + str(uuid.uuid4()).replace('-', '')

        cls.create_indexes([
            {
                "indexName": cls.structured_index_name,
                "type": "structured",
                "model": "hf/all-MiniLM-L6-v2",
                "allFields": [
                    {"name": "text_field_1", "type": "text"},
                    {"name": "text_field_2", "type": "text"}
                ],
                "tensorFields": ["text_field_1", "text_field_2"]
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
                ],
                "tensorFields": ["title", "image_content", "image_field_1"],
            },
            {
                "indexName": cls.unstructured_image_index_name,
                "type": "unstructured",
                "model": "open_clip/ViT-B-32/openai"
            },
            {
                "indexName": cls.unstructured_index_name,
                "type": "unstructured",
            },
            {
                "indexName": cls.unstructured_index_non_e5,
                "type": "unstructured",
                "model": "hf/all-MiniLM-L6-v2"
            }
        ])
        cls.indexes_to_delete = [cls.structured_index_name, cls.unstructured_index_name, cls.unstructured_index_non_e5]

    def test_embed_single_string(self):
        """Embeds a string. Use add docs and get docs with tensor facets to ensure the vector is correct.
                Checks the basic functionality and response structure"""

        test_cases = [self.structured_index_name, self.unstructured_index_name, self.unstructured_index_non_e5]

        for test_index_name in test_cases:
            with (self.subTest(test_index_name)):
                # Add document
                tensor_fields = ["text_field_1"] if "unstr" in test_index_name else None
                d1 = {
                    "_id": "doc1",
                    "text_field_1": "Jimmy Butler is the GOAT."
                }
                res = self.client.index(test_index_name).add_documents([d1], tensor_fields=tensor_fields)

                # Get doc with tensor facets (for reference vector)
                retrieved_d1 = self.client.index(test_index_name).get_document(
                    document_id="doc1", expose_facets=True)

                # Call embed
                if test_index_name == self.unstructured_index_non_e5:
                    embed_res = self.client.index(test_index_name).embed("Jimmy Butler is the GOAT.", device="cpu")
                else:
                    embed_res = self.client.index(test_index_name).embed("Jimmy Butler is the GOAT.", device="cpu", content_type="document")

                self.assertIn("processingTimeMs", embed_res)
                self.assertEqual(embed_res["content"], "Jimmy Butler is the GOAT.")
                self.assertTrue(np.allclose(embed_res["embeddings"][0], retrieved_d1["_tensor_facets"][0] ["_embedding"], atol=1e-6))


    def test_embed_with_device(self):
        """Embeds a string with device parameter. Use add docs and get docs with tensor facets to ensure the vector is correct.
                        Checks the basic functionality and response structure"""
        test_cases = [self.structured_index_name, self.unstructured_index_name, self.unstructured_index_non_e5]

        for test_index_name in test_cases:
            with (self.subTest(test_index_name)):
                # Add document
                tensor_fields = ["text_field_1"] if "unstr" in test_index_name else None
                d1 = {
                    "_id": "doc1",
                    "text_field_1": "Jimmy Butler is the GOAT."
                }
                res = self.client.index(test_index_name).add_documents([d1], tensor_fields=tensor_fields)

                # Get doc with tensor facets (for reference vector)
                retrieved_d1 = self.client.index(test_index_name).get_document(
                    document_id="doc1", expose_facets=True)

                # Call embed
                if test_index_name == self.unstructured_index_non_e5:
                    embed_res = self.client.index(test_index_name).embed(content="Jimmy Butler is the GOAT.", device="cpu")
                else:
                    embed_res = self.client.index(test_index_name).embed(content="Jimmy Butler is the GOAT.", device="cpu", content_type="document")

                self.assertIn("processingTimeMs", embed_res)
                self.assertEqual(embed_res["content"], "Jimmy Butler is the GOAT.")
                self.assertTrue(np.allclose(embed_res["embeddings"][0], retrieved_d1["_tensor_facets"][0] ["_embedding"], atol=1e-6))

    def test_embed_single_dict(self):
        """Embeds a dict. Use add docs and get docs with tensor facets to ensure the vector is correct.
                        Checks the basic functionality and response structure"""
        test_cases = [self.structured_index_name, self.unstructured_index_name, self.unstructured_index_non_e5]

        for test_index_name in test_cases:
            with (self.subTest(test_index_name)):
                # Add document
                tensor_fields = ["text_field_1"] if "unstr" in test_index_name else None
                d1 = {
                    "_id": "doc1",
                    "text_field_1": "Jimmy Butler is the GOAT."
                }
                res = self.client.index(test_index_name).add_documents([d1], tensor_fields=tensor_fields)

                # Get doc with tensor facets (for reference vector)
                retrieved_d1 = self.client.index(test_index_name).get_document(
                    document_id="doc1", expose_facets=True)

                # Call embed
                if test_index_name == self.unstructured_index_non_e5:
                    embed_res = self.client.index(test_index_name).embed(content={"Jimmy Butler is the GOAT.": 1})
                else:
                    embed_res = self.client.index(test_index_name).embed(content={"Jimmy Butler is the GOAT.": 1}, content_type="document")

                self.assertIn("processingTimeMs", embed_res)
                self.assertEqual(embed_res["content"], {"Jimmy Butler is the GOAT.": 1})
                self.assertTrue(np.allclose(embed_res["embeddings"][0], retrieved_d1["_tensor_facets"][0] ["_embedding"], atol=1e-6))

    def test_embed_list_content(self):
        """Embeds a list with string and dict. Use add docs and get docs with tensor facets to ensure the vector is correct.
                                Checks the basic functionality and response structure"""
        test_cases = [self.structured_index_name, self.unstructured_index_name, self.unstructured_index_non_e5]

        for test_index_name in test_cases:
            with (self.subTest(test_index_name)):
                # Add document
                tensor_fields = ["text_field_1"] if "unstr" in test_index_name else None
                d1 = {
                    "_id": "doc1",
                    "text_field_1": "Jimmy Butler is the GOAT."
                }
                d2 = {
                    "_id": "doc2",
                    "text_field_1": "Alex Caruso is the GOAT."
                }
                res = self.client.index(test_index_name).add_documents([d1, d2], tensor_fields=tensor_fields)

                # Get doc with tensor facets (for reference vector)
                retrieved_docs = self.client.index(test_index_name).get_documents(
                    document_ids=["doc1", "doc2"], expose_facets=True)

                # Call embed
                if test_index_name == self.unstructured_index_non_e5:
                    embed_res = self.client.index(test_index_name).embed(
                        content=[{"Jimmy Butler is the GOAT.": 1}, "Alex Caruso is the GOAT."],
                    )
                else:
                    embed_res = self.client.index(test_index_name).embed(
                        content=[{"Jimmy Butler is the GOAT.": 1}, "Alex Caruso is the GOAT."],
                        content_type="document"
                    )

                self.assertIn("processingTimeMs", embed_res)
                self.assertEqual(embed_res["content"], [{"Jimmy Butler is the GOAT.": 1}, "Alex Caruso is the GOAT."])
                self.assertTrue(
                    np.allclose(embed_res["embeddings"][0], retrieved_docs["results"][0]["_tensor_facets"][0]["_embedding"], atol=1e-6))
                self.assertTrue(
                    np.allclose(embed_res["embeddings"][1], retrieved_docs["results"][1]["_tensor_facets"][0]["_embedding"], atol=1e-6))

    def test_embed_documents_for_private_images(self):
        """Both image_download_headers and media_download_headers work in embed for private images."""
        content = ["https://d2k91vq0avo7lq.cloudfront.net/ai_hippo_realistic_small.png",
                   "https://d2k91vq0avo7lq.cloudfront.net/ai_hippo_realistic_small"]
        kwargs_list = [
            {"media_download_headers": {"marqo_media_header": "media_header_test_key"}},
            {"image_download_headers": {"marqo_media_header": "media_header_test_key"}}
        ]
        for index_name in [self.structured_image_index_name, self.unstructured_image_index_name]:
            for kwargs in kwargs_list:
                with self.subTest(f"{index_name} - {kwargs}"):
                    res = self.client.index(index_name).embed(
                        content,
                        **kwargs
                    )
                    self.assertEqual(2, len(res["embeddings"]))

    def test_invalidArgError_is_raised_when_embed_a_private_image(self):
        content = "https://d2k91vq0avo7lq.cloudfront.net/ai_hippo_realistic_small"
        for index_name in [self.structured_image_index_name, self.unstructured_image_index_name]:
            with self.subTest(f"{index_name}"):
                with self.assertRaises(MarqoWebError) as e:
                    self.client.index(index_name).embed(content)
                self.assertIn("Error downloading media file", str(e.exception))

    def test_proper_error_if_both_imageDownloadHeaders_and_mediaDownloadHeaders_are_provided(self):
        """Test that an error is raised if both imageDownloadHeaders and mediaDownloadHeaders are provided."""
        for index_name in [self.unstructured_image_index_name, self.structured_image_index_name]:
            with self.assertRaises(MarqoWebError) as cm:
                res = self.client.index(index_name).embed(
                    "test",
                    image_download_headers={"marqo_media_header": "media_header_test_key"},
                    media_download_headers={"marqo_media_header": "media_header_test_key"}
                )
                self.assertIn("Cannot set both imageDownloadHeaders and mediaDownloadHeaders.",
                              str(cm.exception.message))

