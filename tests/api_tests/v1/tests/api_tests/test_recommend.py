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
                    "model": "hf/all-MiniLM-L6-v2",
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
                    "model": "hf/all-MiniLM-L6-v2",
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

    def test_recommender_rerankDepth(self):
        """Test that rerank_depth affects hit count and handles edge cases based on expected behavior."""
        docs = [
            {"_id": "doc_0", "title": "Project Overview",
             "content": "Summary of the project's goals and deliverables."},
            {"_id": "doc_1", "title": "Team Roles", "content": "Descriptions of each team member's responsibilities."},
            {"_id": "doc_2", "title": "Timeline", "content": "Key milestones and deadlines for the project."},
            {"_id": "doc_3", "title": "Budget Estimate", "content": "Projected costs and resource allocation."},
            {"_id": "doc_4", "title": "Tech Stack", "content": "Overview of technologies and tools being used."},
            {"_id": "doc_5", "title": "Risk Assessment", "content": "Potential risks and mitigation strategies."},
            {"_id": "doc_6", "title": "Client Feedback", "content": "Summary of feedback received from stakeholders."},
            {"_id": "doc_7", "title": "Testing Plan", "content": "Details on testing strategies and coverage."},
            {"_id": "doc_8", "title": "Deployment Guide",
             "content": "Steps and procedures for deploying the application."},
            {"_id": "doc_9", "title": "Post-Mortem",
             "content": "Analysis of what went well and areas for improvement."},
        ]

        for index_name in [self.structured_index_name, self.unstructured_index_name]:
            with self.subTest(index_name=index_name):
                tensor_fields = ["title", "content"] if index_name == self.unstructured_index_name else None
                searchable_attributes = ["title"]

                add_docs_results = self.client.index(index_name).add_documents(
                    docs, tensor_fields=tensor_fields
                )
                if add_docs_results["errors"]:
                    raise Exception(f"Failed to add documents to index {index_name}")

                # Case 1: result_count < rerank_depth → limit is respected
                with self.subTest(case="result_count_less_than_rerank_depth"):
                    res = self.client.index(index_name).recommend(
                        documents=["doc_0", "doc_1"], limit=3, offset=0, rerank_depth=5
                    )
                    self.assertEqual(len(res["hits"]), 3)

                # Case 2: offset > rerank_depth — offset + limit is higher, result must be present
                with self.subTest(case="offset_beyond_rerank_depth"):
                    res = self.client.index(index_name).recommend(
                        documents=["doc_0", "doc_1"], limit=1, offset=3, rerank_depth=2
                    )
                    self.assertEqual(len(res["hits"]), 1)

                # Case 3: offset + result_count <= rerank_depth → return all requested hits
                with self.subTest(case="offset_within_rerank_depth"):
                    res = self.client.index(index_name).recommend(
                        documents=["doc_0", "doc_1"], limit=2, offset=2, rerank_depth=5
                    )
                    self.assertEqual(len(res["hits"]), 2)

                # Case 4: rerank_depth < result_count → result_count overrides rerank_depth
                with self.subTest(case="result_count_exceeds_rerank_depth"):
                    res = self.client.index(index_name).recommend(
                        documents=["doc_0", "doc_1"], limit=5, offset=0, rerank_depth=3
                    )
                    self.assertEqual(len(res["hits"]), 5)

                # Case 5: ef_search < rerank_depth → ef_search limits rerank pool
                with self.subTest(case="ef_search_limits_rerank_pool"):
                    res = self.client.index(index_name).recommend(
                        documents=["doc_0", "doc_1"], limit=10, offset=0, rerank_depth=5, ef_search=3,
                        searchable_attributes=['title']
                    )
                    self.assertLessEqual(len(res["hits"]), 3)

                # Case 6: rerank_depth is negative → should raise error
                with self.subTest(case="invalid_negative_rerank_depth"):
                    with self.assertRaises(MarqoWebError):
                        self.client.index(index_name).recommend(
                            documents=["doc_0", "doc_1"], limit=10, offset=0, rerank_depth=-1
                        )