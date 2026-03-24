"""Integration tests for _id IN filter on semi-structured indexes.

Uses a random model to avoid real inference overhead. Tests with 1000 documents.
"""
import os
from unittest import mock

from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_index import Model, SemiStructuredMarqoIndex
from marqo.exceptions import InvalidArgumentError
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import SearchMethod
from tests.integ_tests.marqo_test import MarqoTestCase


class TestIdInFilterSemiStructured(MarqoTestCase):
    """Tests for _id IN filter on semi-structured (unstructured) indexes with 1000 documents."""

    NUM_DOCS = 1000

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        unstructured_index_request = cls.unstructured_marqo_index_request(
            model=Model(name='random/small', text_query_prefix='', text_chunk_prefix=''),
        )

        cls.indexes = cls.create_indexes([unstructured_index_request])
        cls.index = cls.indexes[0]

        # Verify it's semi-structured
        assert isinstance(cls.index, SemiStructuredMarqoIndex), \
            f"Expected SemiStructuredMarqoIndex, got {type(cls.index)}"

    def setUp(self) -> None:
        super().setUp()
        self.device_patcher = mock.patch.dict(os.environ, {"MARQO_BEST_AVAILABLE_DEVICE": "cpu"})
        self.device_patcher.start()

        # Add 1000 documents
        batch_size = 64
        for start in range(0, self.NUM_DOCS, batch_size):
            end = min(start + batch_size, self.NUM_DOCS)
            docs = [
                {"_id": f"doc_{i}", "title": f"product {i}", "category": f"cat_{i % 10}"}
                for i in range(start, end)
            ]
            self.add_documents(
                config=self.config,
                add_docs_params=AddDocsParams(
                    index_name=self.index.name,
                    docs=docs,
                    tensor_fields=["title"],
                )
            )

    def tearDown(self) -> None:
        super().tearDown()
        self.device_patcher.stop()

    def test_id_in_tensor_search(self):
        """_id IN filter works with TENSOR search."""
        target_ids = {"doc_0", "doc_50", "doc_999"}
        filter_str = "_id IN (doc_0, doc_50, doc_999)"

        res = tensor_search.search(
            config=self.config, index_name=self.index.name,
            text="product", result_count=10,
            filter=filter_str, search_method=SearchMethod.TENSOR
        )

        result_ids = {hit["_id"] for hit in res["hits"]}
        self.assertEqual(target_ids, result_ids)

    def test_id_in_lexical_search(self):
        """_id IN filter works with LEXICAL search."""
        target_ids = {"doc_0", "doc_50", "doc_999"}
        filter_str = "_id IN (doc_0, doc_50, doc_999)"

        res = tensor_search.search(
            config=self.config, index_name=self.index.name,
            text="product", result_count=10,
            filter=filter_str, search_method=SearchMethod.LEXICAL
        )

        result_ids = {hit["_id"] for hit in res["hits"]}
        self.assertEqual(target_ids, result_ids)

    def test_id_in_hybrid_search(self):
        """_id IN filter works with HYBRID search."""
        target_ids = {"doc_0", "doc_50", "doc_999"}
        filter_str = "_id IN (doc_0, doc_50, doc_999)"

        res = tensor_search.search(
            config=self.config, index_name=self.index.name,
            text="product", result_count=10,
            filter=filter_str, search_method=SearchMethod.HYBRID
        )

        result_ids = {hit["_id"] for hit in res["hits"]}
        self.assertEqual(target_ids, result_ids)

    def test_not_id_in(self):
        """NOT _id IN excludes the specified IDs from results."""
        exclude_ids = {f"doc_{i}" for i in range(self.NUM_DOCS - 3, self.NUM_DOCS)}
        filter_str = f"NOT _id IN ({', '.join(exclude_ids)})"

        res = tensor_search.search(
            config=self.config, index_name=self.index.name,
            text="product", result_count=self.NUM_DOCS,
            filter=filter_str, search_method=SearchMethod.TENSOR
        )

        result_ids = {hit["_id"] for hit in res["hits"]}
        # None of the excluded IDs should appear
        self.assertEqual(set(), result_ids & exclude_ids)
        # All other docs should be present
        self.assertEqual(self.NUM_DOCS - len(exclude_ids), len(result_ids))

    def test_id_in_combined_with_equality_filter(self):
        """_id IN combined with AND equality filter on another field."""
        # doc_0 has category=cat_0, doc_10 has category=cat_0, doc_1 has category=cat_1
        filter_str = "_id IN (doc_0, doc_1, doc_10) AND category:cat_0"

        res = tensor_search.search(
            config=self.config, index_name=self.index.name,
            text="product", result_count=10,
            filter=filter_str, search_method=SearchMethod.TENSOR
        )

        result_ids = {hit["_id"] for hit in res["hits"]}
        # Only doc_0 and doc_10 have category=cat_0
        self.assertEqual({"doc_0", "doc_10"}, result_ids)

    def test_id_in_large_list_1000(self):
        """_id IN with all 1000 IDs works without error."""
        all_ids = [f"doc_{i}" for i in range(self.NUM_DOCS)]
        filter_str = "_id IN (" + ", ".join(all_ids) + ")"

        res = tensor_search.search(
            config=self.config, index_name=self.index.name,
            text="product", result_count=self.NUM_DOCS,
            filter=filter_str, search_method=SearchMethod.TENSOR
        )

        self.assertEqual(self.NUM_DOCS, len(res["hits"]))

    def test_id_in_single_id(self):
        """_id IN with a single ID returns exactly one result."""
        res = tensor_search.search(
            config=self.config, index_name=self.index.name,
            text="product", result_count=10,
            filter="_id IN (doc_42)", search_method=SearchMethod.TENSOR
        )

        self.assertEqual(1, len(res["hits"]))
        self.assertEqual("doc_42", res["hits"][0]["_id"])

    def test_id_in_nonexistent_ids(self):
        """_id IN with nonexistent IDs returns 0 hits."""
        res = tensor_search.search(
            config=self.config, index_name=self.index.name,
            text="product", result_count=10,
            filter="_id IN (nonexistent_1, nonexistent_2)", search_method=SearchMethod.TENSOR
        )

        self.assertEqual(0, len(res["hits"]))

    def test_id_in_empty(self):
        """_id IN () returns 0 hits."""
        res = tensor_search.search(
            config=self.config, index_name=self.index.name,
            text="product", result_count=10,
            filter="_id IN ()", search_method=SearchMethod.TENSOR
        )

        self.assertEqual(0, len(res["hits"]))

    def test_non_id_field_in_raises_error(self):
        """IN on non-_id fields raises InvalidArgumentError on semi-structured indexes."""
        with self.assertRaises(InvalidArgumentError) as cm:
            tensor_search.search(
                config=self.config, index_name=self.index.name,
                text="product", result_count=10,
                filter="category IN (cat_0, cat_1)", search_method=SearchMethod.TENSOR
            )

        self.assertIn("only supported for the '_id' field", str(cm.exception))
