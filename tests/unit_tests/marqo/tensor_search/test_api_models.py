import unittest
from marqo.tensor_search.models.api_models import SearchQuery
from marqo.tensor_search.enums import SearchMethod


class TestSearchQueryModel(unittest.TestCase):
    def test_default_values(self):
        """Test default values for SearchQuery model"""
        query = SearchQuery()
        self.assertIsNone(query.q)
        self.assertIsNone(query.searchableAttributes)
        self.assertEqual(query.searchMethod, SearchMethod.TENSOR)
        self.assertEqual(query.limit, 10)
        self.assertEqual(query.offset, 0)
        self.assertIsNone(query.rerankDepth)
        self.assertIsNone(query.efSearch)
        self.assertIsNone(query.approximate)
        self.assertIsNone(query.approximateThreshold)
        self.assertTrue(query.showHighlights)
        self.assertIsNone(query.reRanker)
        self.assertIsNone(query.filter)
        self.assertIsNone(query.attributesToRetrieve)
    
    def test_approximate_threshold_parameter(self):
        """Test setting approximateThreshold parameter"""
        query = SearchQuery(approximateThreshold=0.85)
        self.assertEqual(query.approximateThreshold, 0.85)
        
    def test_setting_all_params(self):
        """Test setting all parameters including approximateThreshold"""
        query = SearchQuery(
            q="test query",
            searchableAttributes=["field1", "field2"],
            searchMethod=SearchMethod.HYBRID,
            limit=20,
            offset=5,
            rerankDepth=30,
            efSearch=100,
            approximate=True,
            approximateThreshold=0.9,
            showHighlights=False,
            reRanker="some-reranker",
            filter="field1:value",
            attributesToRetrieve=["field1", "field3"]
        )
        
        self.assertEqual(query.q, "test query")
        self.assertEqual(query.searchableAttributes, ["field1", "field2"])
        self.assertEqual(query.searchMethod, SearchMethod.HYBRID)
        self.assertEqual(query.limit, 20)
        self.assertEqual(query.offset, 5)
        self.assertEqual(query.rerankDepth, 30)
        self.assertEqual(query.efSearch, 100)
        self.assertTrue(query.approximate)
        self.assertEqual(query.approximateThreshold, 0.9)
        self.assertFalse(query.showHighlights)
        self.assertEqual(query.reRanker, "some-reranker")
        self.assertEqual(query.filter, "field1:value")
        self.assertEqual(query.attributesToRetrieve, ["field1", "field3"])


if __name__ == "__main__":
    unittest.main() 