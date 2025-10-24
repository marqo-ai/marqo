from marqo.vespa.models.query_result import MarqoFields, RootFields
from unittest import TestCase


class TestMarqoFields(TestCase):
    def test_marqo_fields_with_all_fields(self):
        """Test MarqoFields with all fields populated"""
        data = {
            "sortCandidates": 10,
            "relevantCandidates": 5,
            "probeCandidates": 3
        }
        fields = MarqoFields(**data)
        
        self.assertEqual(fields.sort_candidates, 10)
        self.assertEqual(fields.relevant_candidates, 5)
        self.assertEqual(fields.probe_candidates, 3)
    
    def test_marqo_fields_with_sort_candidates(self):
        """Test MarqoFields with only some fields populated"""
        data = {
            "sortCandidates": 10,
        }
        fields = MarqoFields(**data)
        
        self.assertEqual(fields.sort_candidates, 10)
        self.assertIsNone(fields.relevant_candidates)
        self.assertIsNone(fields.probe_candidates)
    
    def test_marqo_fields_empty(self):
        """Test MarqoFields with empty data"""
        data = dict()
        fields = MarqoFields(**data)
        self.assertIsNone(fields.sort_candidates)
        self.assertIsNone(fields.relevant_candidates)
        self.assertIsNone(fields.probe_candidates)

    def test_no_marqo_fields(self):
        """Test QueryResult without MarqoFields"""
        data = {
            "totalCount": 100
        }
        result = RootFields(**data)

        self.assertIsNone(result.marqo_fields)
        self.assertEqual(100, result.total_count)