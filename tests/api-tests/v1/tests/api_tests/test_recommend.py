import uuid

from marqo.client import Client
from marqo.enums import InterpolationMethod
from marqo.errors import MarqoWebError

from tests.marqo_test import MarqoTestCase


class TestRecommend(MarqoTestCase):
    structured_index_name = "structured_index" + str(uuid.uuid4()).replace('-', '')
    unstructured_index_name = "unstructured_index" + str(uuid.uuid4()).replace('-', '')

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.create_indexes(
            [
                {
                    "indexName": cls.structured_index_name,
                    "type": "structured",
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "allFields": [
                        {"name": "title", "type": "text", "features": ["filter", "lexical_search"]},
                        {"name": "content", "type": "text", "features": ["filter", "lexical_search"]},
                        {"name": "tags", "type": "array<text>", "features": ["filter"]},
                        {"name": "int_filter_field_1", "type": "int", "features": ["filter", "score_modifier"]}
                    ],
                    "tensorFields": ["title", "content"],
                },
                {
                    "indexName": cls.unstructured_index_name,
                    "type": "unstructured",
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                }
            ]
        )

        cls.indexes_to_delete = [cls.structured_index_name, cls.unstructured_index_name]

    def test_recommend_defaults(self):
        """
        Test recommend with only required fields provided
        """
        docs = [
            {
                "_id": "1",
                "title": "Red orchid",
                "tags": ["flower", "orchid"],
            },
            {
                "_id": "2",
                "title": "Red rose",
                "tags": ["flower"],
            },
            {
                "_id": "3",
                "title": "Europe",
                "tags": ["continent"],
            },
        ]

        for index_name in [self.structured_index_name, self.unstructured_index_name]:
            with self.subTest(index_name):
                tensor_fields = ["title"] if index_name == self.unstructured_index_name else None
                add_docs_results = self.client.index(index_name).add_documents(docs, tensor_fields=tensor_fields)

                if add_docs_results["errors"]:
                    raise Exception(f"Failed to add documents to index {index_name}")

                res = self.client.index(index_name).recommend(
                    documents=['1', '2']
                )

                ids = [doc["_id"] for doc in res["hits"]]
                self.assertEqual(set(ids), {"3"})

    def test_recommend_allFields(self):
        """
        Test recommend with all fields provided
        """
        docs = [
            {
                "_id": "1",
                "title": "Red orchid",
                "content": "flower",
            },
            {
                "_id": "2",
                "title": "Red rose",
                "content": "flower"
            },
            {
                "_id": "3",
                "title": "Europe",
                "content": "continent",
            }
        ]

        for index_name in [self.structured_index_name, self.unstructured_index_name]:
            with self.subTest(index_name):
                tensor_fields = ["title"] if index_name == self.unstructured_index_name else None
                searchable_attributes = ["title"]
                add_docs_results = self.client.index(index_name).add_documents(docs, tensor_fields=tensor_fields)

                if add_docs_results["errors"]:
                    raise Exception(f"Failed to add documents to index {index_name}")

                res = self.client.index(index_name).recommend(
                    documents=['1', '2'],
                    tensor_fields=["title"],
                    interpolation_method=InterpolationMethod.SLERP,
                    exclude_input_documents=True,
                    limit=10,
                    offset=0,
                    ef_search=100,
                    approximate=True,
                    searchable_attributes=searchable_attributes,
                    show_highlights=True,
                    filter_string='content:(continent)',
                    attributes_to_retrieve=["title"],
                    score_modifiers={
                        "multiply_score_by":
                            [
                                {
                                    "field_name": "int_filter_field_1",
                                    "weight": 1
                                }
                            ]
                    }
                )
                ids = [doc["_id"] for doc in res["hits"]]
                self.assertEqual(set(ids), {"3"})

    def test_recommender_documentsWithoutEmbeddings(self):
        """
        Test recommend with documents that do not have embeddings
        """
        docs = [
            {
                "_id": "1",
                "title": "Red orchid",
                "tags": ["flower", "orchid"],
            },
            {
                "_id": "2",
                "title": "Red rose",
                "tags": ["flower"],
                "content": "test"
            },
            {
                "_id": "3",
                "title": "Europe",
                "tags": ["continent"],
            },
        ]

        for index_name in [self.structured_index_name, self.unstructured_index_name]:
            with self.subTest(index_name):
                tensor_fields = ["content"] if index_name == self.unstructured_index_name else None
                add_docs_results = self.client.index(index_name).add_documents(docs, tensor_fields=tensor_fields)

                if add_docs_results["errors"]:
                    raise Exception(f"Failed to add documents to index {index_name}")

                with self.assertRaises(MarqoWebError) as e:
                    self.client.index(index_name).recommend(
                        documents=['1', '2', '3'], tensor_fields=["content"]
                    )
                self.assertIn("1, 3", str(e.exception))

    def test_recommender_structuredDocumentsNoTensorFields(self):
        """Test to ensure that an error is raised when invalid tensor fields are provided for a structured index"""
        docs = [
            {
                "_id": "1",
                "title": "Red orchid",
                "tags": ["flower", "orchid"],
            },
            {
                "_id": "2",
                "title": "Red rose",
                "tags": ["flower"],
                "content": "test"
            },
            {
                "_id": "3",
                "title": "Europe",
                "tags": ["continent"],
            },
        ]

        index_name = self.structured_index_name
        add_docs_results = self.client.index(index_name).add_documents(docs)
        if add_docs_results["errors"]:
            raise Exception(f"Failed to add documents to index {index_name}")

        with self.assertRaises(MarqoWebError) as e:
            self.client.index(index_name).recommend(
                documents=['1', '2', '3'], tensor_fields=["void"]
            )
        self.assertIn("Available tensor fields: title, content", str(e.exception))