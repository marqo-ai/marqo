"""Integration tests for _id IN filter on semi-structured indexes.

Uses a random model to avoid real inference overhead. Adds 10,000 documents once
in setUpClass, then tests filter queries including 10,000-ID lists.
"""
import os
from unittest import mock

import pytest

from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.facets_parameters import FacetsParameters, FieldFacetsConfiguration
from marqo.core.models.hybrid_parameters import RetrievalMethod, RankingMethod, HybridParameters
from marqo.core.models.marqo_index import Model, SemiStructuredMarqoIndex
from marqo.exceptions import InvalidArgumentError
from marqo.settings.settings import Settings, get_settings
from marqo.tensor_search import tensor_search
from marqo.tensor_search.enums import SearchMethod
from tests.integ_tests.marqo_test import MarqoTestCase


class TestIdInFilterSemiStructured(MarqoTestCase):
    """Tests for _id IN filter on semi-structured (unstructured) indexes.

    Inserts 10,000 documents once at class level. Tests large IN lists up to 10,000 IDs.
    """

    NUM_DOCS = 10000
    ALL_IDS = {f"doc_{i}" for i in range(10000)}

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        unstructured_index_request = cls.unstructured_marqo_index_request(
            model=Model(name='random/small', text_query_prefix='', text_chunk_prefix=''),
        )

        cls.indexes = cls.create_indexes([unstructured_index_request])
        cls.index = cls.indexes[0]

        assert isinstance(cls.index, SemiStructuredMarqoIndex), \
            f"Expected SemiStructuredMarqoIndex, got {type(cls.index)}"

        # Add 10,000 documents once for all tests
        batch_size = 64
        for start in range(0, cls.NUM_DOCS, batch_size):
            end = min(start + batch_size, cls.NUM_DOCS)
            docs = [
                {"_id": f"doc_{i}", "title": f"product {i}", "category": f"cat_{i % 10}"}
                for i in range(start, end)
            ]
            cls.add_documents(
                config=cls.config,
                add_docs_params=AddDocsParams(
                    index_name=cls.index.name,
                    docs=docs,
                    tensor_fields=["title"],
                )
            )

    def setUp(self) -> None:
        # Skip base class setUp which clears all documents.
        # Data is shared read-only across tests and inserted once in setUpClass.
        pass

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
        """NOT _id IN excludes the specified IDs and returns all remaining docs.

        Uses a pool of 20 docs, excludes 5, verifies the exact remaining 15.
        """
        pool_ids = sorted([f"doc_{i}" for i in range(20)])
        exclude_ids = sorted([f"doc_{i}" for i in range(15, 20)])
        expected_ids = set(pool_ids) - set(exclude_ids)

        pool_filter = "_id IN (" + ", ".join(pool_ids) + ")"
        exclude_filter = f"NOT _id IN ({', '.join(exclude_ids)})"
        filter_str = f"{pool_filter} AND {exclude_filter}"

        res = tensor_search.search(
            config=self.config, index_name=self.index.name,
            text="product", result_count=20,
            filter=filter_str, search_method=SearchMethod.TENSOR
        )

        result_ids = {hit["_id"] for hit in res["hits"]}
        self.assertEqual(expected_ids, result_ids)

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
        self.assertEqual({"doc_0", "doc_10"}, result_ids)

    def test_id_in_large_list_10000_ids(self):
        """_id IN with all 10,000 IDs works without error.

        MARQO_MAX_SEARCH_LIMIT caps results at 1,000, so we verify the query succeeds
        and all returned IDs are from our doc set.
        """
        all_ids = [f"doc_{i}" for i in range(self.NUM_DOCS)]
        filter_str = "_id IN (" + ", ".join(all_ids) + ")"

        res = tensor_search.search(
            config=self.config, index_name=self.index.name,
            text="product", result_count=1000,
            filter=filter_str, search_method=SearchMethod.TENSOR
        )

        self.assertEqual(1000, len(res["hits"]))
        result_ids = {hit["_id"] for hit in res["hits"]}
        self.assertTrue(result_ids.issubset(self.ALL_IDS))

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

    def test_id_in_exceeds_max_limit_raises_error(self):
        """_id IN with more IDs than MARQO_MAX_IN_FILTER_IDS raises InvalidArgumentError."""
        max_ids = 5
        ids = [f"doc_{i}" for i in range(max_ids + 1)]
        filter_str = "_id IN (" + ", ".join(ids) + ")"

        with mock.patch("marqo.settings.settings._settings", Settings(marqo_max_in_filter_ids=max_ids)):
            with self.assertRaises(InvalidArgumentError) as cm:
                tensor_search.search(
                    config=self.config, index_name=self.index.name,
                    text="product", result_count=10,
                    filter=filter_str, search_method=SearchMethod.TENSOR
                )

            self.assertIn("MARQO_MAX_IN_FILTER_IDS", str(cm.exception))
            self.assertIn(str(max_ids), str(cm.exception))

    def test_id_in_at_max_limit_succeeds(self):
        """_id IN with exactly MARQO_MAX_IN_FILTER_IDS values succeeds."""
        max_ids = 5
        ids = [f"doc_{i}" for i in range(max_ids)]
        filter_str = "_id IN (" + ", ".join(ids) + ")"

        with mock.patch("marqo.settings.settings._settings", Settings(marqo_max_in_filter_ids=max_ids)):
            res = tensor_search.search(
                config=self.config, index_name=self.index.name,
                text="product", result_count=10,
                filter=filter_str, search_method=SearchMethod.TENSOR
            )

            self.assertEqual(max_ids, len(res["hits"]))

    def test_id_in_default_limit_is_10000(self):
        """Default MARQO_MAX_IN_FILTER_IDS is 10,000."""
        self.assertEqual(10000, get_settings().marqo_max_in_filter_ids)

    def test_id_in_and_or_filter(self):
        """_id IN (...) AND (_id:doc0 OR _id:doc1) returns the intersection."""
        # IN list has doc_0, doc_1, doc_2. The OR clause matches doc_0 and doc_1.
        # AND semantics: result = {doc_0, doc_1}
        filter_str = "_id IN (doc_0, doc_1, doc_2) AND (_id:doc_0 OR _id:doc_1)"

        res = tensor_search.search(
            config=self.config, index_name=self.index.name,
            text="product", result_count=10,
            filter=filter_str, search_method=SearchMethod.TENSOR
        )

        result_ids = {hit["_id"] for hit in res["hits"]}
        self.assertEqual({"doc_0", "doc_1"}, result_ids)

    def test_id_in_or_or_filter(self):
        """_id IN (...) OR (_id:doc0 OR _id:doc1) returns the union."""
        # IN list has doc_2 and doc_3. OR clause adds doc_0 and doc_1.
        # OR semantics: result = {doc_0, doc_1, doc_2, doc_3}
        filter_str = "_id IN (doc_2, doc_3) OR (_id:doc_0 OR _id:doc_1)"

        res = tensor_search.search(
            config=self.config, index_name=self.index.name,
            text="product", result_count=10,
            filter=filter_str, search_method=SearchMethod.TENSOR
        )

        result_ids = {hit["_id"] for hit in res["hits"]}
        self.assertEqual({"doc_0", "doc_1", "doc_2", "doc_3"}, result_ids)

    def test_id_in_and_not_or_filter(self):
        """_id IN (...) AND NOT (_id:doc0 OR _id:doc1) excludes the OR'd IDs."""
        # IN list has doc_0 through doc_3. NOT clause excludes doc_0 and doc_1.
        # Result: {doc_2, doc_3}
        filter_str = "_id IN (doc_0, doc_1, doc_2, doc_3) AND NOT (_id:doc_0 OR _id:doc_1)"

        res = tensor_search.search(
            config=self.config, index_name=self.index.name,
            text="product", result_count=10,
            filter=filter_str, search_method=SearchMethod.TENSOR
        )

        result_ids = {hit["_id"] for hit in res["hits"]}
        self.assertEqual({"doc_2", "doc_3"}, result_ids)

    @pytest.mark.skip_for_multinode
    def test_id_in_with_facets(self):
        """_id IN filter works with the facets feature.

        Documents: doc_0 (cat_0), doc_1 (cat_1), doc_10 (cat_0).
        Expects category facets: {cat_0: 2, cat_1: 1}.
        """
        filter_str = "_id IN (doc_0, doc_1, doc_10)"
        facets = FacetsParameters(fields={"category": FieldFacetsConfiguration(type="string")})

        res = tensor_search.search(
            config=self.config, index_name=self.index.name,
            text="product", result_count=10,
            filter=filter_str,
            facets=facets,
            search_method=SearchMethod.HYBRID,
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF
            )
        )

        result_ids = {hit["_id"] for hit in res["hits"]}
        self.assertEqual({"doc_0", "doc_1", "doc_10"}, result_ids)
        self.assertDictEqual(
            {"category": {"cat_0": {"count": 2}, "cat_1": {"count": 1}}},
            res["facets"]
        )

    @pytest.mark.skip_for_multinode
    def test_id_in_with_facets_exclude_terms(self):
        """_id IN filter works with facets and excludeTerms on the IN statement.

        The filter restricts hits to doc_0 and doc_10 (both cat_0).
        excludeTerms removes the _id IN restriction from the facet scope,
        so facets count all docs matching category:cat_0 across the full index
        (1000 docs — one per every 10 of the 10,000 inserted docs).
        """
        filter_str = "_id IN (doc_0, doc_1, doc_10) AND category:cat_0"
        facets = FacetsParameters(fields={"category": FieldFacetsConfiguration(
            type="string", excludeTerms=["_id IN (doc_0,doc_1,doc_10)"]
        )})

        res = tensor_search.search(
            config=self.config, index_name=self.index.name,
            text="product", result_count=10,
            filter=filter_str,
            facets=facets,
            search_method=SearchMethod.HYBRID,
            hybrid_parameters=HybridParameters(
                retrievalMethod=RetrievalMethod.Disjunction,
                rankingMethod=RankingMethod.RRF
            )
        )

        # Hits are filtered by the full filter (IN list AND category:cat_0)
        result_ids = {hit["_id"] for hit in res["hits"]}
        self.assertEqual({"doc_0", "doc_10"}, result_ids)

        # Facets are computed without the _id IN restriction (excludeTerms removed it),
        # so the facet scope is just category:cat_0 across all 10,000 docs → 1000 matches.
        self.assertDictEqual(
            {"category": {"cat_0": {"count": 1000}}},
            res["facets"]
        )
