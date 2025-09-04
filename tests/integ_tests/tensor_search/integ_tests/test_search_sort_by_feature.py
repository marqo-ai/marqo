import json
import pytest
import semver
from unittest.mock import patch, MagicMock

from fastapi.exceptions import RequestValidationError

from marqo.core.exceptions import UnsupportedFeatureError
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index import MarqoIndex
from marqo.tensor_search.api import search
from marqo.tensor_search.enums import SearchMethod
from tests.integ_tests.marqo_test import MarqoTestCase


class TestSearchSortByFeature(MarqoTestCase):
    """
    This test class is designed to test the sort_by API of Marqo search API.
    Functional tests are in other classes.
    """

    def test_sort_by_is_blocked_if_the_index_version_is_prior_to_2_22(self):
        """
        Test that sort_by is blocked if the index version is prior to 2.22.
        """
        # Create an index with a version prior to 2.21
        mock_index = MagicMock(spec=MarqoIndex)
        mock_index.parsed_marqo_version.return_value = semver.VersionInfo.parse('2.21.0')  # Version prior to 2.22
        mock_index.name = "test_index"

        with (patch("marqo.tensor_search.tensor_search.index_meta_cache.get_index", return_value=mock_index)
              as mock_get_index):
            # Attempt to use sort_by on the index
            with self.assertRaises(UnsupportedFeatureError) as e:
                search(
                    index_name=mock_index.name,
                    marqo_config=self.config,
                    device="cpu",
                    search_query_dict={
                        "q": "test",
                        "searchMethod": SearchMethod.HYBRID,
                        "hybridParameters": {
                            "retrievalMethod": "disjunction",
                            "rankingMethod": "rrf",
                            "alpha": 0.5,
                        },
                        "sortBy": {
                            "fields": [
                                {
                                    "field_name": "sort_field_1",
                                    "order": "asc",
                                    "missing": "first"
                                }
                            ]
                        }
                    }
                )

            self.assertIn(
                "The 'sortBy' features is only supported for unstructured indexes created with Marqo "
                "version 2.22.0 or later",
                str(e.exception)
            )

    def test_sort_by_is_block_if_the_search_is_on_a_structured_index(self):
        mock_index = MagicMock(spec=MarqoIndex)
        mock_index.parsed_marqo_version.return_value = semver.VersionInfo.parse('2.22.0')
        mock_index.name = "test_index"
        mock_index.type="structured" # Type set to 2.21

        with (patch("marqo.tensor_search.tensor_search.index_meta_cache.get_index", return_value=mock_index)
              as mock_get_index):
            # Attempt to use sort_by on the index
            with self.assertRaises(UnsupportedFeatureError) as e:
                search(
                    index_name=mock_index.name,
                    marqo_config=self.config,
                    device="cpu",
                    search_query_dict={
                        "q": "test",
                        "searchMethod": SearchMethod.HYBRID,
                        "hybridParameters": {
                            "retrievalMethod": "disjunction",
                            "rankingMethod": "rrf",
                            "alpha": 0.5,
                        },
                        "sortBy": {
                            "fields": [
                                {
                                    "field_name": "sort_field_1",
                                    "order": "asc",
                                    "missing": "first"
                                }
                            ]
                        }
                    }
                )

            self.assertIn(
                "The 'sortBy' features is only supported for unstructured indexes "
                "created with Marqo version 2.22.0 or later",
                str(e.exception)
            )

    def test_sort_by_is_blocked_if_the_index_is_a_legacy_index(self):
        mock_index = MagicMock(spec=MarqoIndex)
        mock_index.parsed_marqo_version.return_value = semver.VersionInfo.parse('2.12.0')
        mock_index.name = "test_index"
        mock_index.type="unstructured"

        with (patch("marqo.tensor_search.tensor_search.index_meta_cache.get_index", return_value=mock_index)
              as mock_get_index):
            # Attempt to use sort_by on the index
            with self.assertRaises(UnsupportedFeatureError) as e:
                search(
                    index_name=mock_index.name,
                    marqo_config=self.config,
                    device="cpu",
                    search_query_dict={
                        "q": "test",
                        "searchMethod": SearchMethod.HYBRID,
                        "hybridParameters": {
                            "retrievalMethod": "disjunction",
                            "rankingMethod": "rrf",
                            "alpha": 0.5,
                        },
                        "sortBy": {
                            "fields": [
                                {
                                    "field_name": "sort_field_1",
                                    "order": "asc",
                                    "missing": "first"
                                }
                            ]
                        }
                    }
                )

            self.assertIn(
                "The 'sortBy' features is only supported for unstructured indexes created with Marqo version "
                "2.22.0 or later",
                str(e.exception)
            )


class TestSearchSortByFeatureSort1Field(MarqoTestCase):
    """
    This test class is designed to test the sorting functionality of the Marqo search API, with only one sort field.
    We want to solve the following cases:
    1. Sort by 1 single field;
    2. Sort by 1 single field with relevance as a tie-breaker;
    3. Sort by 1 single field with different sort orders (ascending and descending);
    4. Sort by 1 single field with missing values policy (e.g., first, last, or none).
    5. Sort by 1 single field will field name of different types (e.g., string). In this case, the field should be
    treated as a missing field, and the sort order should be applied accordingly.
    6. Sorty by 1 single field but the field never exists in the index.
    7. Test limit, offset, sortDepth, minSortCandidates parameters to ensure they work as expected.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        semi_structured_default_text_index = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all-MiniLM-L6-v2')
        )

        cls.create_indexes([semi_structured_default_text_index])

        cls.index_name = semi_structured_default_text_index.name

        # Documents for TestSearchSortByFeatureSort1Field
        test_sort1field_docs = [
            {"_id": "0", "content": ' '.join([f"content{i}" for i in range(1)]), "sort_field_1": 0.0},  # zero value
            {"_id": "1", "content": ' '.join([f"content{i}" for i in range(4)]), "sort_field_1": 5.3},  # mid value
            {"_id": "2", "content": ' '.join([f"content{i}" for i in range(6)]), "sort_field_1": 10},  # highest value
            {"_id": "3", "content": ' '.join([f"content{i}" for i in range(10)]), "sort_field_1": 3},  # tie value for relevance tiebreak
            {"_id": "4", "content": ' '.join([f"content{i}" for i in range(7)]), "sort_field_1": 3},  # tie value for relevance tiebreak
            {"_id": "5", "content": ' '.join([f"content{i}" for i in range(8)]), "sort_field_1": "invalid"},  # wrong type treated as missing
            {"_id": "6", "content": ' '.join([f"content{i}" for i in range(3)]), "sort_field_1": ["test"]},  # wrong type treated as missing
            {"_id": "7", "content": ' '.join([f"content{i}" for i in range(9)])},  # missing field entirely
            {"_id": "8", "content": ' '.join([f"content{i}" for i in range(2)]), "sort_field_1": -1},  # negative value
            {"_id": "9", "content": ' '.join([f"content{i}" for i in range(5)]), "sort_field_1": 2.5},  # float value
        ]

        _ = cls.add_documents(
            config=cls.config,
            add_docs_params=AddDocsParams(
                docs=test_sort1field_docs,
                index_name=semi_structured_default_text_index.name,
                documents=test_sort1field_docs,
                tensor_fields=['content'],
            )
        )

        expected_normal_search_order = [
            "3", "7", "5", "4", "2", "9", "1", "6", "8", "0"
        ]

        # Check if the documents are indexed correctly
        normal_search_res = [r["_id"] for r in cls._help_sort_function()["hits"]]
        if normal_search_res != expected_normal_search_order:
            raise RuntimeError(
                f"Expected normal search order to be {expected_normal_search_order}, "
                f"but got {normal_search_res}"
            )

    def setUp(self):
        """Ensure documents are not changed before each test."""
        if 10 !=self.monitoring.get_index_stats_by_name(self.index_name).number_of_documents:
            raise RuntimeError(
                f"Expected 10 documents in index {self.index_name} for sorting tests"
            )

    def tearDown(self):
        """Ensure documents are not changed after each test."""
        if 10 !=self.monitoring.get_index_stats_by_name(self.index_name).number_of_documents:
            raise RuntimeError(
                f"Expected 10 documents in index {self.index_name} for sorting tests"
            )

    @classmethod
    def _help_sort_function(cls, query: Optional[str] = ' '.join([f"content{i}" for i in range(10)]),
                            sort_by: Optional[dict] = None, limit=10, offset=0,
                            hybrid_parameters: Optional[dict] = None
                            ) -> dict:
        if hybrid_parameters is None:
            hybrid_parameters = {
                "retrievalMethod": "disjunction",
                "rankingMethod": "rrf",
                "alpha": 0.5,
            }

        return json.loads(search(
            index_name=cls.index_name,
            marqo_config=cls.config,
            device="cpu",
            search_query_dict={
                "q": query,
                "searchMethod": SearchMethod.HYBRID,
                "hybridParameters": hybrid_parameters,
                "sortBy": sort_by,
                "limit": limit,
                "offset": offset
            }
        ).body.decode('utf-8'))

    def test_sort_by_is_blocked_by_tensor_search(self):
        with self.assertRaises(RequestValidationError) as context:
            _ = search(
                index_name=self.index_name,
                marqo_config=self.config,
                device="cpu",
                search_query_dict={
                    "q": "machine learning artificial intelligence algorithms",
                    "searchMethod": SearchMethod.TENSOR,
                    "sortBy": {
                        "fields": [{"fieldName": "sort_field_1"}],
                    }
                }
            )
        self.assertIn("sortBy can only be provided for", str(context.exception.errors()))

    def test_sort_by_is_blocked_by_lexical_search(self):
        with self.assertRaises(RequestValidationError) as context:
            _ = search(
                index_name=self.index_name,
                marqo_config=self.config,
                device="cpu",
                search_query_dict={
                    "q": "machine learning artificial intelligence algorithms",
                    "searchMethod": SearchMethod.LEXICAL,
                    "sortBy": {
                        "fields": [{"fieldName": "sort_field_1"}],
                    }
                }
            )
        self.assertIn("sortBy can only be provided for", str(context.exception.errors()))

    def test_sort_by_can_not_used_with_score_modifiers(self):
        """Ensure that sort_by cannot be used with score modifiers."""
        with self.assertRaises(RequestValidationError) as context:
            _ = search(
                index_name=self.index_name,
                marqo_config=self.config,
                device="cpu",
                search_query_dict={
                    "q": "machine learning artificial intelligence algorithms",
                    "searchMethod": SearchMethod.HYBRID,
                    "sortBy": {
                        "fields": [{"fieldName": "sort_field_1"}],
                    },
                    "scoreModifiers": {"multiply_score_by": [{"field_name": "sort_field_1", "weight": 2}]}
                }
            )
        self.assertIn(
            "'sortBy' cannot be used with 'scoreModifiers'(global score modifiers)",
                      str(context.exception.errors())
        )

    def test_simple_sort_with_default_settings(self):
        """
        The simple sort test check based on default values:
        - Sort by a single field (sort_field_1).
        - Sort order is descending by default.
        - No missing values policy is specified, so the default is 'last'.
        So the results should be in the following order:
            [
                "2", "1", # numeric values in descending order
                "3", "4", # tie values sorted by relevance, with _id 3 coming before _id 4
                "9", "0", "8", # descending order of numeric values
                "7", "5", "6" # missing fields sorted by relevance, with _id 7 coming before _id 5 and _id 6
            ]
        """
        sort_by = {
            "fields": [
                {
                    "field_name": "sort_field_1",
                }
            ]
        }
        res = self._help_sort_function(sort_by=sort_by)
        self.assertEqual(10, res["_sortCandidates"])
        hits = res["hits"]
        self.assertEqual(10, len(hits))
        ids = [hit["_id"] for hit in hits]
        self.assertEqual(
            ['2', '1', '3', '4', '9', '0', '8', '7', '5', '6'],
            ids
        )

    def test_hybrid_sorting_with_lexical_tensor_search_with_defaults(self):
        sort_by = {
            "fields": [
                {
                    "field_name": "sort_field_1",
                }
            ]
        }

        hybrid_search_parameters = {
            "retrievalMethod": "lexical",
            "rankingMethod": "tensor",
        }

        res = self._help_sort_function(sort_by=sort_by, hybrid_parameters=hybrid_search_parameters)
        self.assertEqual(10, res["_sortCandidates"])
        hits = res["hits"]
        self.assertEqual(10, len(hits))
        ids = [hit["_id"] for hit in hits]
        self.assertEqual(
            ['2', '1', '3', '4', '9', '0', '8', '7', '5', '6'],
            ids
        )

    def test_hybrid_sorting_with_lexical_lexical_search_with_defaults(self):
        sort_by = {
            "fields": [
                {
                    "field_name": "sort_field_1",
                }
            ]
        }

        hybrid_search_parameters = {
            "retrievalMethod": "lexical",
            "rankingMethod": "lexical",
        }

        res = self._help_sort_function(sort_by=sort_by, hybrid_parameters=hybrid_search_parameters)
        self.assertEqual(10, res["_sortCandidates"])
        hits = res["hits"]
        self.assertEqual(10, len(hits))
        ids = [hit["_id"] for hit in hits]
        self.assertEqual(
            ['2', '1', '3', '4', '9', '0', '8', '7', '5', '6'],
            ids
        )

    def test_hybrid_sorting_with_tensor_lexical_search_with_defaults(self):
        sort_by = {
            "fields": [
                {
                    "field_name": "sort_field_1",
                }
            ]
        }

        hybrid_search_parameters = {
            "retrievalMethod": "tensor",
            "rankingMethod": "lexical",
        }

        res = self._help_sort_function(sort_by=sort_by, hybrid_parameters=hybrid_search_parameters)
        self.assertEqual(10, res["_sortCandidates"])
        hits = res["hits"]
        self.assertEqual(10, len(hits))
        ids = [hit["_id"] for hit in hits]
        self.assertEqual(
            ['2', '1', '3', '4', '9', '0', '8', '7', '5', '6'],
            ids
        )

    def test_hybrid_sorting_with_tensor_tensor_search_with_defaults(self):
        sort_by = {
            "fields": [
                {
                    "field_name": "sort_field_1",
                }
            ]
        }

        hybrid_search_parameters = {
            "retrievalMethod": "tensor",
            "rankingMethod": "tensor",
        }

        res = self._help_sort_function(sort_by=sort_by, hybrid_parameters=hybrid_search_parameters)
        self.assertEqual(10, res["_sortCandidates"])
        hits = res["hits"]
        self.assertEqual(10, len(hits))
        ids = [hit["_id"] for hit in hits]
        self.assertEqual(
            ['2', '1', '3', '4', '9', '0', '8', '7', '5', '6'],
            ids
        )

    def test_simple_sort_non_default_parameters(self):
        """
        The simple sort test check based on default values:
        - Sort by a single field (sort_field_1).
        - Sort order is ascending.
        - Missing values policy is set to 'first'.

        Expected results:
            [
                "7", "5", "6", # Missing fields should come first, with relevance as a tie-breaker
                "8", "0", "9", # ascending order of numeric values
                "3", "4",  # Tie values should be sorted by relevance, with _id 3 coming before _id 4
                "1", "2" # ascending order of the highest values
            ]
        """
        sort_by = {
            "fields": [
                {
                    "field_name": "sort_field_1",
                    "order": "asc",  # Ascending order
                    "missing": "first"
                }
            ]
        }
        res = self._help_sort_function(sort_by=sort_by)
        self.assertEqual(10, res["_sortCandidates"])
        hits = res["hits"]
        self.assertEqual(10, len(hits))
        ids = [hit["_id"] for hit in hits]
        self.assertEqual(
            ["7", "5", "6", "8", "0", "9", "3", "4", "1", "2"],
            ids
        )

    def test_sort_desc_missing_first(self):
        """Test sorting descending with missing values first."""
        sort_by = {
            "fields": [
                {
                    "field_name": "sort_field_1",
                    "order": "desc",
                    "missing": "first"
                }
            ]
        }
        res = self._help_sort_function(sort_by=sort_by)
        hits = res["hits"]
        ids = [hit["_id"] for hit in hits]
        # Missing first: ["7", "5", "6"], then desc: ["2", "1", "3", "4", "9", "0", "8"]
        self.assertEqual(
            ["7", "5", "6", "2", "1", "3", "4", "9", "0", "8"],
            ids
        )

    def test_sort_asc_missing_last(self):
        """Test sorting ascending with missing values last."""
        sort_by = {
            "fields": [
                {
                    "field_name": "sort_field_1",
                    "order": "asc",
                    "missing": "last"
                }
            ]
        }
        res = self._help_sort_function(sort_by=sort_by)
        hits = res["hits"]
        ids = [hit["_id"] for hit in hits]
        # Asc: ["8", "0", "9", "3", "4", "1", "2"], then missing last: ["7", "5", "6"]
        self.assertEqual(
            ["8", "0", "9", "3", "4", "1", "2", "7", "5", "6"],
            ids
        )

    @pytest.mark.skip_for_multinode(
        "The lexical score can differ between nodes so the results may not consistently match"
    )
    def test_sort_by_when_fields_does_not_exist(self):
        """
        Test sorting by a field that does not exist in the index.
        The expected behavior is all the documents should be returned in the same order as if no sort was applied,
        however the _score field should be different as we use normalized relevance scores during sorting, even if
        the sort field does not exist in the index.
        """
        sort_by = {
            "fields": [
                {
                    "field_name": "non_existent_field",
                    "order": "asc",  # Ascending order
                    "missing": "last"  # Missing values should come last
                }
            ]
        }

        res = self._help_sort_function(sort_by=sort_by)
        regular_res = self._help_sort_function()
        self.assertEqual(len(regular_res["hits"]), len(res["hits"]))
        self.assertEqual(
            [hit["_id"] for hit in regular_res["hits"]],
            [hit["_id"] for hit in res["hits"]]
        )
        for i in range(len(regular_res["hits"])):
            for field in regular_res["hits"][i].keys():
                if field != "_score":
                    self.assertEqual(
                        regular_res["hits"][i][field],
                        res["hits"][i][field],
                        f"Regular hits: {regular_res['hits'][i]}, Sorted hits: {res['hits'][i]},"
                    )
                else:
                    self.assertNotEqual(
                        regular_res["hits"][i][field],
                        res["hits"][i][field],
                        f"Regular hits: {regular_res['hits'][i]}, Sorted hits: {res['hits'][i]},"
                    )

    def test_sort_depth_parameter(self):
        """
        Test the sort depth parameter to ensure it works as expected.
        The sort depth should limit the number of documents considered for sorting, with the rest being the
        same as if no sort was applied.

        Expected results:
            - Before sort: ['3', '7', '5', '4', '2', '9', '1', '6', '8', '0']
            - After sort with sortDepth=6:
            [
                '2', # Highest value
                '3', '4', # Tie values sorted by relevance, with _id 3 coming before _id 4
                '9', # Sorted be descending order of numeric values
                '7', '5', # Missing fields last, sorted by relevance
                '1', '6', '8', '0' # Unsorted documents after sort depth limit
            ]
        """
        sort_by = {
            "fields": [
                {
                    "field_name": "sort_field_1",
                    "order": "desc",
                    "missing": "last"
                }
            ],
            "sortDepth": 6  # Limit the sort depth to 6
        }
        res = self._help_sort_function(sort_by=sort_by)
        self.assertEqual(10, res["_sortCandidates"])
        hits = res["hits"]
        self.assertEqual(10, len(hits))
        ids = [hit["_id"] for hit in hits]
        self.assertEqual(
            ['2', '3', '4', '9', '7', '5', '1', '6', '8', '0'],
            ids
        )

    def test_sort_by_with_limit_and_offset(self):
        """
        Test the sort by functionality with limit and offset parameters. We also include the sort depth parameter here
        for a more comprehensive test.
        Expected results:
            - Before sort: ['3', '7', '5', '4', '2', '9', '1', '6', '8', '0']
            - After sort with sortDepth=4:
            [
                '3', '4', # Tie values sorted by relevance, with _id 3 coming before _id 4
                '7', '5', # Missing fields last, sorted by relevance
                '2', '9', '1', '6', '8', '0' # Unsorted documents after sort depth limit
            ]
            And we trim the results based on the limit and offset parameters.
        """
        limit = 5
        offset = 2

        sort_by = {
            "fields": [
                {
                    "field_name": "sort_field_1",
                    "order": "desc",
                    "missing": "last"
                }
            ],
            "sortDepth": 4,  # Limit the sort depth to 6
            "minSortCandidates": max(10, limit+offset)  # Ensure we have enough candidates to sort
        }

        res = self._help_sort_function(sort_by=sort_by, limit=limit, offset=offset)
        self.assertEqual(10, res["_sortCandidates"])
        hits = res["hits"]
        ids = [hit["_id"] for hit in hits]
        self.assertEqual(
            # Adjust the expected ids based on offset and limit
            ['3', '4', '7', '5', '2', '9', '1', '6', '8', '0'][offset:offset + limit],
            ids
        )

    def test_small_sort_limit_without_specifying_min_sort_candidates(self):
        """
        Test the case where the sort limit is smaller than the number of documents,
        and sortCandidates is not specified.
        In this case, the sort candidates is defaulted to be max(3 * limit, limit + offset).

        # We get the top 6 hits and only return the top 2 hits by sort order, so we return ['2', '3'].
        """
        sort_by = {
            "fields": [
                {
                    "field_name": "sort_field_1",
                    "order": "desc",
                    "missing": "last"
                }
            ]
        }

        res = self._help_sort_function(sort_by=sort_by, limit=2, offset=0)

        self.assertEqual(6, res["_sortCandidates"])  # Default is max(3 * limit, limit + offset)
        hits = res["hits"]
        self.assertEqual(2, len(hits))
        ids = [hit["_id"] for hit in hits]
        self.assertEqual(
            ['2', '3'],
            ids
        )

    def test_sort_depth_edge_cases(self):
        """Test sortDepth edge cases: equal to hit size, larger than hit size, and 1."""
        sort_by_base = {
            "fields": [
                {
                    "field_name": "sort_field_1",
                    "order": "desc",
                    "missing": "last"
                }
            ]
        }

        # Test sortDepth equal to total documents (10)
        sort_by_equal = {**sort_by_base, "sortDepth": 10}
        res = self._help_sort_function(sort_by=sort_by_equal)
        self.assertEqual(10, res["_sortCandidates"])
        # Should be fully sorted
        ids = [hit["_id"] for hit in res["hits"]]
        self.assertEqual(['2', '1', '3', '4', '9', '0', '8', '7', '5', '6'], ids)

        # Test sortDepth larger than total documents (15)
        sort_by_larger = {**sort_by_base, "sortDepth": 15}
        res = self._help_sort_function(sort_by=sort_by_larger)
        self.assertEqual(10, res["_sortCandidates"])
        # Should be fully sorted (same as above)
        ids = [hit["_id"] for hit in res["hits"]]
        self.assertEqual(['2', '1', '3', '4', '9', '0', '8', '7', '5', '6'], ids)

        # Test sortDepth of 1 (minimal)
        sort_by_minimal = {**sort_by_base, "sortDepth": 1}
        res = self._help_sort_function(sort_by=sort_by_minimal)
        self.assertEqual(10, res["_sortCandidates"])
        # Only first document sorted, rest in original relevance order
        ids = [hit["_id"] for hit in res["hits"]]
        self.assertEqual(['3', '7', '5', '4', '2', '9', '1', '6', '8', '0'], ids)

    def test_sort_depth_null_handling(self):
        """Test sortDepth when not specified (null/None)."""
        sort_by = {
            "fields": [
                {
                    "field_name": "sort_field_1",
                    "order": "desc",
                    "missing": "last"
                }
            ]
            # No sortDepth specified - should default to sorting all
        }

        res = self._help_sort_function(sort_by=sort_by)
        self.assertEqual(10, res["_sortCandidates"])
        # Should be fully sorted since no depth limit
        ids = [hit["_id"] for hit in res["hits"]]
        self.assertEqual(['2', '1', '3', '4', '9', '0', '8', '7', '5', '6'], ids)


class TestSearchSortByFeatureSort2Fields(MarqoTestCase):
    """
    Test sorting functionality of the Marqo search API when sorting by two fields.
    Primary sort on sort_field_1, secondary sort on sort_field_2.
    """
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # create a fresh index for two-field sorting tests
        idx = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all-MiniLM-L6-v2')
        )
        cls.create_indexes([idx])
        cls.index_name = idx.name

        # Documents with two sort fields
        docs = [
            {"_id": "0", "content": ' '.join([f"content{i}" for i in range(1)]), "sort_field_1": 0.0,
             "sort_field_2": 10},
            {"_id": "1", "content": ' '.join([f"content{i}" for i in range(4)]), "sort_field_1": 5.3,
             "sort_field_2": 5},
            {"_id": "2", "content": ' '.join([f"content{i}" for i in range(6)]), "sort_field_1": 10, "sort_field_2": 0},
            {"_id": "3", "content": ' '.join([f"content{i}" for i in range(10)]), "sort_field_1": 3, "sort_field_2": 5},
            {"_id": "4", "content": ' '.join([f"content{i}" for i in range(7)]), "sort_field_1": 3, "sort_field_2": 7},
            {"_id": "5", "content": ' '.join([f"content{i}" for i in range(8)]), "sort_field_1": "invalid",
             "sort_field_2": 1},
            {"_id": "6", "content": ' '.join([f"content{i}" for i in range(3)]), "sort_field_1": ["test"],
             "sort_field_2": 2},
            {"_id": "7", "content": ' '.join([f"content{i}" for i in range(9)])},
            {"_id": "8", "content": ' '.join([f"content{i}" for i in range(2)]), "sort_field_1": -1,
             "sort_field_2": -5},
            {"_id": "9", "content": ' '.join([f"content{i}" for i in range(5)]), "sort_field_1": 2.5,
             "sort_field_2": "invalid"},
        ]

        # index documents
        cls.add_documents(
            config=cls.config,
            add_docs_params=AddDocsParams(
                docs=docs,
                index_name=cls.index_name,
                documents=docs,
                tensor_fields=['content'],
            )
        )

        # verify indexing order without explicit sort (primary by relevance)
        expected = ["3", "7", "5", "4", "2", "9", "1", "6", "8", "0"]
        actual = [r["_id"] for r in cls._help_sort_function()["hits"]]
        if actual != expected:
            raise RuntimeError(f"Unexpected default relevance order: {actual}")

    @classmethod
    def _help_sort_function(cls, query: Optional[str] = ' '.join([f"content{i}" for i in range(10)]),
                            sort_by: Optional[dict] = None, limit=10, offset=0) -> dict:
        return json.loads(search(
            index_name=cls.index_name,
            marqo_config=cls.config,
            device="cpu",
            search_query_dict={
                "q": query,
                "searchMethod": SearchMethod.HYBRID,
                "hybridParameters": {
                    "retrievalMethod": "disjunction",
                    "rankingMethod": "rrf",
                    "alpha": 0.5,
                },
                "sortBy": sort_by,
                "limit": limit,
                "offset": offset
            }
        ).body.decode('utf-8'))

    def setUp(self):
        # ensure count unchanged before each test
        count = self.monitoring.get_index_stats_by_name(self.index_name).number_of_documents
        self.assertEqual(count, 10, f"Expected 10 documents, found {count}")

    def tearDown(self):
        # ensure count unchanged after each test
        count = self.monitoring.get_index_stats_by_name(self.index_name).number_of_documents
        self.assertEqual(count, 10, f"Expected 10 documents, found {count}")

    def test_simple_sort_two_fields_default_settings(self):
        """
        Expected results:
            [
                "2", "1", # primary sort_field_1 desc
                "4", "3", # primary sort_field_1 tie, sorted by sort_field_2 desc
                "9", "0", "8", # primary sort_field_1 desc,
                "6", "5", # missing primary first, sorted by sort_field_2 desc
                "7", # missing both fields
            ]
        """
        sort_by = {
            "fields": [
                {"field_name": "sort_field_1"},
                {"field_name": "sort_field_2"}
            ]
        }
        res = self._help_sort_function(sort_by=sort_by)
        ids = [h["_id"] for h in res["hits"]]
        # primary desc on field1, secondary desc on field2
        self.assertEqual(['2', '1', '4', '3', '9', '0', '8', '6', '5', '7'], ids)

    def test_simple_sort_two_fields_non_default_parameters(self):  # asc, missing first
        sort_by = {
            "fields": [
                {"field_name": "sort_field_1", "order": "asc", "missing": "first"},
                {"field_name": "sort_field_2", "order": "asc", "missing": "first"}
            ]
        }
        res = self._help_sort_function(sort_by=sort_by)
        ids = [h["_id"] for h in res["hits"]]
        # missing primary first (7,5,6), then field1 asc, then field2 asc tie-break
        self.assertEqual(['7', '5', '6', '8', '0', '9', '3', '4', '1', '2'], ids)

    @pytest.mark.skip_for_multinode(
        "The lexical score can differ between nodes so the results may not consistently match")
    def test_sort_by_when_fields_do_not_exist_two_fields(self):
        """
        Sorting with non-existent fields should return documents in the same order as if no sort was applied, with
        the relevance score being different due to normalization.
        """
        sort_by = {
            "fields": [
                {"field_name": "no_field_1", "order": "asc", "missing": "last"},
                {"field_name": "no_field_2", "order": "asc", "missing": "last"}
            ]
        }
        # ensure ordering (excluding score) matches relevance-only order
        base = self._help_sort_function()
        res = self._help_sort_function(sort_by=sort_by)
        for i in range(len(base["hits"])):
            for k in base["hits"][i]:
                if k != '_score':
                    self.assertEqual(base["hits"][i][k], res["hits"][i][k],
                                     f"Field {k} does not match for hit {i}")
                else:
                    self.assertNotEqual(base["hits"][i][k], res["hits"][i][k])

    def test_sort_two_fields_first_fixed_second_desc_first(self):
        """Test 2 fields: first field fixed (desc/last), second field desc/first."""
        sort_by = {
            "fields": [
                {"field_name": "sort_field_1", "order": "desc", "missing": "last"},
                {"field_name": "sort_field_2", "order": "desc", "missing": "first"}
            ]
        }
        res = self._help_sort_function(sort_by=sort_by)
        ids = [h["_id"] for h in res["hits"]]
        self.assertEqual(['2', '1', '4', '3', '9', '0', '8', '7', '6', '5'], ids)

    def test_sort_two_fields_first_fixed_second_asc_last(self):
        """Test 2 fields: first field fixed (desc/last), second field asc/last."""
        sort_by = {
            "fields": [
                {"field_name": "sort_field_1", "order": "desc", "missing": "last"},
                {"field_name": "sort_field_2", "order": "asc", "missing": "last"}
            ]
        }
        res = self._help_sort_function(sort_by=sort_by)
        ids = [h["_id"] for h in res["hits"]]
        # Primary desc on field1, secondary asc on field2 with missing last
        # Expected: ['2', '1', '3', '4', '9', '0', '8', '5', '6', '7']
        self.assertEqual(['2', '1', '3', '4', '9', '0', '8', '5', '6', '7'], ids)

    def test_sort_two_fields_first_fixed_second_desc_last(self):
        """Test 2 fields: first field fixed (desc/last), second field desc/last."""
        sort_by = {
            "fields": [
                {"field_name": "sort_field_1", "order": "desc", "missing": "last"},
                {"field_name": "sort_field_2", "order": "desc", "missing": "last"}
            ]
        }
        res = self._help_sort_function(sort_by=sort_by)
        ids = [h["_id"] for h in res["hits"]]
        # Primary desc on field1, secondary desc on field2 with missing last
        # Expected: ['2', '1', '4', '3', '9', '0', '8', '6', '5', '7']
        self.assertEqual(['2', '1', '4', '3', '9', '0', '8', '6', '5', '7'], ids)

    def test_sort_two_fields_first_fixed_second_asc_first(self):
        """Test 2 fields: first field fixed (desc/last), second field asc/first."""
        sort_by = {
            "fields": [
                {"field_name": "sort_field_1", "order": "desc", "missing": "last"},
                {"field_name": "sort_field_2", "order": "asc", "missing": "first"}
            ]
        }
        res = self._help_sort_function(sort_by=sort_by)
        ids = [h["_id"] for h in res["hits"]]
        # Primary desc on field1, secondary asc on field2 with missing first
        # Expected: ['2', '1', '3', '4', '9', '0', '8', '7', '5', '6']
        self.assertEqual(['2', '1', '3', '4', '9', '0', '8', '7', '5', '6'], ids)


class TestSearchSortByFeatureSort3Fields(MarqoTestCase):
    """
    Test sorting functionality of the Marqo search API when sorting by three fields.
    Primary: sort_field_1, Secondary: sort_field_2, Tertiary: sort_field_3.
    """
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # create a fresh index for three-field sorting tests
        idx = cls.unstructured_marqo_index_request(
            model=Model(name='hf/all-MiniLM-L6-v2')
        )
        cls.create_indexes([idx])
        cls.index_name = idx.name

        # Documents with three sort fields (types mixed to test missing/invalid handling)
        docs = [
            {"_id": "0", "content": "content0", "sort_field_1": 0.0, "sort_field_2": 10, "sort_field_3": 1},
            {"_id": "1", "content": " ".join(f"content{i}" for i in range(4)),
             "sort_field_1": 5.3, "sort_field_2": 5, "sort_field_3": 2},
            {"_id": "2", "content": " ".join(f"content{i}" for i in range(6)),
             "sort_field_1": 10, "sort_field_2": 0, "sort_field_3": 3},
            # Tie on field1=3 and field2=5, broken by field3
            {"_id": "3", "content": " ".join(f"content{i}" for i in range(10)),
             "sort_field_1": 3, "sort_field_2": 5, "sort_field_3": 5},
            {"_id": "4", "content": " ".join(f"content{i}" for i in range(7)),
             "sort_field_1": 3, "sort_field_2": 5, "sort_field_3": 4},
            # Invalid / missing cases
            {"_id": "5", "content": " ".join(f"content{i}" for i in range(8)),
             "sort_field_1": "invalid", "sort_field_2": 1, "sort_field_3": 6},
            {"_id": "6", "content": " ".join(f"content{i}" for i in range(3)),
             "sort_field_1": ["test"], "sort_field_2": 2}, # Missing field 3
            {"_id": "7", "content": " ".join(f"content{i}" for i in range(9))},  # missing all three
            {"_id": "8", "content": " ".join(f"content{i}" for i in range(2)),
             "sort_field_1": -1, "sort_field_2": -5, "sort_field_3": -2},
            {"_id": "9", "content": " ".join(f"content{i}" for i in range(5)),
             "sort_field_1": 2.5, "sort_field_2": "invalid", "sort_field_3": 8},
        ]

        cls.add_documents(
            config=cls.config,
            add_docs_params=AddDocsParams(
                docs=docs,
                index_name=cls.index_name,
                documents=docs,
                tensor_fields=['content'],
            )
        )

        # verify default relevance ordering
        expected_relevance = ["3", "7", "5", "4", "2", "9", "1", "6", "8", "0"]
        actual = [r["_id"] for r in cls._help_sort_function()["hits"]]
        if actual != expected_relevance:
            raise RuntimeError(f"Unexpected default relevance order: {actual}")

    @classmethod
    def _help_sort_function(cls, query: Optional[str] = ' '.join(f"content{i}" for i in range(10)),
                            sort_by: Optional[dict] = None, limit=10, offset=0) -> dict:
        return json.loads(search(
            index_name=cls.index_name,
            marqo_config=cls.config,
            device="cpu",
            search_query_dict={
                "q": query,
                "searchMethod": SearchMethod.HYBRID,
                "hybridParameters": {
                    "retrievalMethod": "disjunction",
                    "rankingMethod": "rrf",
                    "alpha": 0.5,
                },
                "sortBy": sort_by,
                "limit": limit,
                "offset": offset
            }
        ).body.decode('utf-8'))

    def setUp(self):
        count = self.monitoring.get_index_stats_by_name(self.index_name).number_of_documents
        self.assertEqual(count, 10, f"Expected 10 docs, found {count}")

    def tearDown(self):
        count = self.monitoring.get_index_stats_by_name(self.index_name).number_of_documents
        self.assertEqual(count, 10, f"Expected 10 docs, found {count}")

    def test_simple_sort_three_fields_default_settings(self):
        """
        Default: all descending, missing-last.
        Expect:
          ['2','1',      # sort_field_1: 10, 5.3
           '3','4',      # tie 3 → tie-break on field2 then field3  (tie on f1 & f2 → f3: 5>4)
           '9','0','8',  # 2.5, 0, -1
           '6','5',      # missing f1 group, sorted by f2 desc: 2,1
           '7']          # missing f1 & f2 & f3
        """
        sort_by = {
            "fields": [
                {"field_name": "sort_field_1"},
                {"field_name": "sort_field_2"},
                {"field_name": "sort_field_3"}
            ]
        }
        res = self._help_sort_function(sort_by=sort_by)
        ids = [h["_id"] for h in res["hits"]]
        self.assertEqual(['2', '1', '3', '4', '9', '0', '8', '6', '5', '7'], ids)

    def test_simple_sort_three_fields_non_default_parameters(self):
        """
        Ascending + missing-first on all three:
        - missing f1 first (7,5,6), within that: f2 asc → missing f2 (7), 1 (5), 2 (6)
        - then f1 asc: -1(8),0(0),2.5(9),3(3,4),5.3(1),10(2)
        - tie on f1=3: both f2=5 → f3 asc: 4(4) then 5(3)
        """
        sort_by = {
            "fields": [
                {"field_name": "sort_field_1", "order": "asc", "missing": "first"},
                {"field_name": "sort_field_2", "order": "asc", "missing": "first"},
                {"field_name": "sort_field_3", "order": "asc", "missing": "first"}
            ]
        }
        res = self._help_sort_function(sort_by=sort_by)
        ids = [h["_id"] for h in res["hits"]]
        self.assertEqual(
            ['7', '5', '6',  # missing f1 group
             '8', '0', '9',  # then f1=-1,0,2.5
             '4', '3',  # f1=3 tie: f2 same → f3 asc: 4<5
             '1', '2'], # then 5.3,10
            ids
        )

    @pytest.mark.skip_for_multinode("The lexical score can differ between nodes so the results may not consistently match")
    def test_sort_by_when_fields_do_not_exist_three_fields(self):
        """
        Sorting by three non-existent fields should preserve relevance-only order
        (but _score is normalized differently).
        """
        sort_by = {
            "fields": [
                {"field_name": "no1", "order": "asc", "missing": "last"},
                {"field_name": "no2", "order": "asc", "missing": "last"},
                {"field_name": "no3", "order": "asc", "missing": "last"}
            ]
        }
        base = self._help_sort_function()
        res = self._help_sort_function(sort_by=sort_by)
        for i in range(len(base["hits"])):
            for k in base["hits"][i]:
                if k != "_score":
                    self.assertEqual(base["hits"][i][k], res["hits"][i][k],
                                     f"Field {k} mismatch at position {i}")
                else:
                    self.assertNotEqual(base["hits"][i][k], res["hits"][i][k])

    def test_sort_three_fields_first_two_fixed_third_desc_first(self):
        """Test 3 fields: first two fixed (desc/last), third field desc/first."""
        sort_by = {
            "fields": [
                {"field_name": "sort_field_1", "order": "desc", "missing": "last"},
                {"field_name": "sort_field_2", "order": "desc", "missing": "last"},
                {"field_name": "sort_field_3", "order": "desc", "missing": "first"}
            ]
        }
        res = self._help_sort_function(sort_by=sort_by)
        ids = [h["_id"] for h in res["hits"]]
        # Expected based on the test data and sort order
        # Primary: field1 desc, Secondary: field2 desc, Tertiary: field3 desc with missing first
        self.assertEqual(['2', '1', '3', '4', '9', '0', '8', '6', '5', '7'], ids)

    def test_sort_three_fields_first_two_fixed_third_asc_last(self):
        """Test 3 fields: first two fixed (desc/last), third field asc/last."""
        sort_by = {
            "fields": [
                {"field_name": "sort_field_1", "order": "desc", "missing": "last"},
                {"field_name": "sort_field_2", "order": "desc", "missing": "last"},
                {"field_name": "sort_field_3", "order": "asc", "missing": "last"}
            ]
        }
        res = self._help_sort_function(sort_by=sort_by)
        ids = [h["_id"] for h in res["hits"]]
        # Expected: field1 desc, field2 desc, field3 asc with missing last
        self.assertEqual(['2', '1', '4', '3', '9', '0', '8', '6', '5', '7'], ids)

    def test_sort_three_fields_first_two_fixed_third_desc_last(self):
        """Test 3 fields: first two fixed (desc/last), third field desc/last."""
        sort_by = {
            "fields": [
                {"field_name": "sort_field_1", "order": "desc", "missing": "last"},
                {"field_name": "sort_field_2", "order": "desc", "missing": "last"},
                {"field_name": "sort_field_3", "order": "desc", "missing": "last"}
            ]
        }
        res = self._help_sort_function(sort_by=sort_by)
        ids = [h["_id"] for h in res["hits"]]
        # Expected: field1 desc, field2 desc, field3 desc with missing last
        self.assertEqual(['2', '1', '3', '4', '9', '0', '8', '6', '5', '7'], ids)

    def test_sort_three_fields_first_two_fixed_third_asc_first(self):
        """Test 3 fields: first two fixed (desc/last), third field asc/first."""
        sort_by = {
            "fields": [
                {"field_name": "sort_field_1", "order": "desc", "missing": "last"},
                {"field_name": "sort_field_2", "order": "desc", "missing": "last"},
                {"field_name": "sort_field_3", "order": "asc", "missing": "first"}
            ]
        }
        res = self._help_sort_function(sort_by=sort_by)
        ids = [h["_id"] for h in res["hits"]]
        # Expected: field1 desc, field2 desc, field3 asc with missing first
        self.assertEqual(['2', '1', '4', '3', '9', '0', '8', '6', '5', '7'], ids)

    def test_sort_three_fields_relevance_tiebreaker(self):
        """Test relevance tiebreaker when all three sort fields are identical."""
        # Create documents where sort fields have identical values to test relevance tiebreaker
        # This is a corner case where sorting falls back to relevance
        sort_by = {
            "fields": [
                {"field_name": "sort_field_1", "order": "desc", "missing": "last"},
                {"field_name": "sort_field_2", "order": "desc", "missing": "last"},
                {"field_name": "sort_field_3", "order": "desc", "missing": "last"}
            ]
        }

        # For documents with identical sort field values (like 3 and 4 both have field1=3, field2=5),
        # relevance should be the tiebreaker
        res = self._help_sort_function(sort_by=sort_by)
        ids = [h["_id"] for h in res["hits"]]

        # Find documents 3 and 4 in the results - they should be ordered by relevance
        # since they have identical sort field values (field1=3, field2=5)
        pos_3 = ids.index('3')
        pos_4 = ids.index('4')

        # Document 3 should come before 4 due to higher relevance (based on content length)
        self.assertLess(pos_3, pos_4, "Document 3 should come before 4 due to relevance tiebreaker")

    def test_comprehensive_pagination_with_sort(self):
        """Test comprehensive pagination scenarios with different sort configurations."""
        test_cases = [
            {"limit": 3, "offset": 0, "expected_length": 3},
            {"limit": 5, "offset": 2, "expected_length": 5},
            {"limit": 10, "offset": 5, "expected_length": 5},  # Only 5 docs left after offset 5
            {"limit": 2, "offset": 8, "expected_length": 2},   # Last 2 docs
            {"limit": 15, "offset": 0, "expected_length": 10}, # Limit exceeds total docs
        ]

        sort_by = {
            "fields": [
                {"field_name": "sort_field_1", "order": "desc", "missing": "last"},
                {"field_name": "sort_field_2", "order": "asc", "missing": "first"},
                {"field_name": "sort_field_3", "order": "desc", "missing": "last"}
            ]
        }

        # Get the full sorted order first
        full_res = self._help_sort_function(sort_by=sort_by, limit=10, offset=0)
        full_ids = [h["_id"] for h in full_res["hits"]]

        for case in test_cases:
            with self.subTest(case=case):
                res = self._help_sort_function(
                    sort_by=sort_by,
                    limit=case["limit"],
                    offset=case["offset"]
                )

                # Verify correct number of results
                self.assertEqual(len(res["hits"]), case["expected_length"])

                # Verify results match the expected slice of full sorted order
                actual_ids = [h["_id"] for h in res["hits"]]
                expected_ids = full_ids[case["offset"]:case["offset"] + case["limit"]]
                self.assertEqual(actual_ids, expected_ids,
                               f"Pagination failed for limit={case['limit']}, offset={case['offset']}")

    def test_missing_field_in_some_documents(self):
        """Test corner case where sort field exists in some docs but missing in others."""
        # This tests the scenario where field exists in matchfeatures for some docs but not others
        # Using sort_field_3 which is missing in document 7
        sort_by = {
            "fields": [
                {"field_name": "sort_field_3", "order": "desc", "missing": "first"}
            ]
        }

        res = self._help_sort_function(sort_by=sort_by)
        ids = [h["_id"] for h in res["hits"]]

        self.assertEqual([
            '7', '6', # Two missing sort_field_3, sorted by relevance
            '9', '5', '3', '4', '2', '1', '0', '8' # Remaining documents sorted by sort_field_3 desc
        ], ids)

    def test_if_all_fields_missing_relevance_is_the_tie_breaker(self):
        """Test if all sort fields are missing, relevance should be the tiebreaker."""
        sort_by = {
            "fields": [
                {"field_name": "sort_field_void", "order": "desc", "missing": "last"}
            ]
        }

        res = self._help_sort_function(sort_by=sort_by)
        ids = [h["_id"] for h in res["hits"]]

        self.assertEqual(
            ["3", "7", "5", "4", "2", "9", "1", "6", "8", "0"],
            ids,
        )