import uuid

from marqo.client import Client

from tests.marqo_test import MarqoTestCase
from marqo.errors import MarqoWebError

import requests


def get_documents_by_ids_via_get(client, index_name, document_ids, expose_facets=False):
    return client.index(index_name).get_documents_by_ids(
        document_ids=document_ids,
        expose_facets=expose_facets
    )


def get_documents_by_ids_via_post(client, index_name, document_ids, expose_facets=False):
    body = {
        "documentIds": document_ids,
    }
    url = f"{client.base_url}/indexes/{index_name}/documents/get-batch"

    if expose_facets:
        url += f"?expose_facets={expose_facets}"

    return requests.post(url, json=body)


class TestGetDocuments(MarqoTestCase):
    """A class to test the get_documents_by_ids functionality for structured and unstructured indexes."""

    structured_image_index_name = "structured_image_index" + str(uuid.uuid4()).replace('-', '')
    unstructured_image_index_name = "unstructured_image_index" + str(uuid.uuid4()).replace('-', '')

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = Client(**cls.client_settings)

        cls.create_indexes([
            {
                "indexName": cls.structured_image_index_name,
                "type": "structured",
                "model": "open_clip/ViT-B-32/laion2b_s34b_b79k",
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
                "indexName": cls.unstructured_image_index_name,
                "type": "unstructured",
                "model": "open_clip/ViT-B-32/laion2b_s34b_b79k",
                "treatUrlsAndPointersAsMedia": True
            }
        ])

        cls.indexes_to_delete = [cls.structured_image_index_name,cls.unstructured_image_index_name]

    def test_get_and_post_received_same_results_when_expose_facets_false(self):
        documents = [
            {
                "_id": "1",
                "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
                "text_field_1": "hello world"
            },
            {
                "_id": "2",
                "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
                "text_field_1": "This is a test"
            },
            {
                "_id": "3",
                "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image3.jpg",
                "text_field_1": "Another test"
            }
        ]

        for index_name in [self.structured_image_index_name, self.unstructured_image_index_name]:
            with self.subTest(index_name=index_name):
                self.client.index(index_name).add_documents(documents)
                document_ids = ["1", "2", "3"]

                get_response = get_documents_by_ids_via_get(self.client, index_name, document_ids)
                post_response = get_documents_by_ids_via_post(self.client, index_name, document_ids).json()

                self.assertEqual(get_response, post_response)

    def test_get_and_post_received_same_results_when_expose_facets_true(self):
        documents = [
            {
                "_id": "1",
                "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
                "text_field_1": "hello world"
            },
            {
                "_id": "2",
                "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
                "text_field_1": "This is a test"
            },
            {
                "_id": "3",
                "image_field_1": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image3.jpg",
                "text_field_1": "Another test"
            }
        ]

        for index_name in [self.structured_image_index_name, self.unstructured_image_index_name]:
            with self.subTest(index_name=index_name):
                self.client.index(index_name).add_documents(documents)
                document_ids = ["1", "2", "3"]

                get_response = get_documents_by_ids_via_get(self.client, index_name, document_ids, expose_facets=True)
                post_response = get_documents_by_ids_via_post(
                    self.client,
                    index_name, document_ids, expose_facets=True).json()

                self.assertEqual(get_response, post_response)

    def test_get_documents_by_ids_via_post_return_correct_error(self):
        test_cases = [
            ({"documentIds": []}, 422, "Empty documentIds"),
            ({"documentIds": None}, 422, "documentIds must be a list"),
            ({"documentIds": {}}, 422, "documentIds must be a list"),
        ]

        for body, expected_status_code, msg in test_cases:
            with self.subTest(msg=msg):
                url = f"{self.client.base_url}/indexes/{self.structured_image_index_name}/documents/get-batch"
                response = requests.post(url, json=body)
                self.assertEqual(response.status_code, expected_status_code)
                self.assertIn("value is not a valid list", response.text)