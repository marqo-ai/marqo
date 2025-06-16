import json

from tests.integ_tests.marqo_test import MarqoTestCase
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_index import *
from marqo.tensor_search.api import search
from marqo.tensor_search.enums import SearchMethod


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
                    "sortDepth": 10,
                    "minSortCandidates": 10
                }
            ]
        }
        for _ in range(10):
            # We run it several times to ensure that the results are consistent
            res = self._help_sort_function(sort_by=sort_by)
            self.assertEqual(10, res["_sortByCandidates"])
            hits = res["hits"]
            self.assertEqual(10, len(hits))
            ids = [hit["_id"] for hit in hits]
            self.assertNotEqual(
                ['2', '1', '3', '4', '9', '0', '8', '7', '5', '6'],
                ids
            )
            # Currently sort is not implemented in Vespa, so it is same as the relevance order.
            self.assertEqual([
                "3", "7", "5", "4", "2", "9", "1", "6", "8", "0"
            ], ids)
