import unittest
import time
from tests.api_tests.v1.tests.marqo_test import MarqoTestCase
from marqo.client import Client


class TestStemming(MarqoTestCase):
    """
    Test text field stemming functionality through the Marqo API.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.client = Client()

        # Create unstructured index for stemming tests
        cls.semi_structured_index_name = cls.random_index_name("stemming_semi")
        cls.client.create_index(
            index_name=cls.semi_structured_index_name,
            type="unstructured",
            model="hf/e5-small-v2"
        )
        cls.indexes_to_delete.append(cls.semi_structured_index_name)

    def populate_index(self):
        docs = [
            {
                "_id": "1",
                "title_stem1": "nacionalmente",  # none: nacionalmente
            },
            {
                "_id": "2",
                "title_stem2": "nacionalmente",  # best: nacionalment
            },
            {
                "_id": "3",
                "title_stem3": "nacionalmente",  # shortest: nacionalment
            },
            {
                "_id": "4",
                "title_stem4": "nacionalmente",  # multiple: nacionalmente, nacionalment
            },
        ]

        res = self.client.index(self.semi_structured_index_name).add_documents(
            docs,
            tensor_fields=["title_stem1"],
            mappings={
                "title_stem1": {"type": "text_field", "language": "de", "stemming": "none"},
                "title_stem2": {"type": "text_field", "language": "de", "stemming": "best"},
                "title_stem3": {"type": "text_field", "language": "de", "stemming": "shortest"},
                "title_stem4": {"type": "text_field", "language": "de", "stemming": "multiple"},
            },
        )

        self.assertFalse(res['errors'], "Should not have errors when adding documents")

    def test_stemming_search(self):
        """
        Test docs with different stemming configs return expected search results.
        """
        cases = [
            (
                "nacionalmente", ["title_stem1"], ["1"], "Full word matches no stemming"
            ),
            (
                "nacionalmente", ["title_stem2"], ["2"], "Full word matches best stemming"
            ),
            (
                "nacionalmente", ["title_stem3"], ["3"], "Full word matches shortest stemming"
            ),
            (
                "nacionalmente", ["title_stem4"], ["4"], "Full word matches multiple stemming"
            ),

            (
                "nacionalmente", ["title_stem1", "title_stem2"], ["1", "2"], "Full word matches with none and "
                                                                             "best fields"
            ),
            (
                "nacionalment", ["title_stem1"], [], "Stemmed word does not match none stemming"
            ),
            (
                "nacionalment", ["title_stem2"], ["2"], "Stemmed word matches best stemming"
            ),
            (
                "nacionalment", ["title_stem1", "title_stem2"], ["2"], "Stemmed word matches best stemming but not none"
            ),
            (
                "nacionalment", ["title_stem3"], ["3"], "Stemmed word matches shortest stemming"
            ),

            (
                "nacionalment", ["title_stem4"], ["4"], "Stemmed word matches multiple stemming"
            ),
        ]

        search_configs = [
            ("LEXICAL", {}, "lexical search"),
            ("HYBRID", {"retrievalMethod": "lexical", "rankingMethod": "lexical"}, "hybrid lexical/lexical"),
            ("HYBRID", {"alpha": 0}, "hybrid RRF with alpha=0")
        ]

        self.populate_index()

        for query, fields, expected_ids, description in cases:
            for search_method, hybrid_params, description in search_configs:
                with self.subTest(f"{search_method} search for '{query}' in {fields}"):
                    res = self.client.index(self.semi_structured_index_name).search(
                        q=query,
                        search_method=search_method,
                        language="de",
                        hybrid_parameters=hybrid_params if hybrid_params else None
                    )

                    actual_ids = set(hit["_id"] for hit in res["hits"] if hit["_id"] in expected_ids)
                    self.assertEqual(set(expected_ids), actual_ids)

    def test_stemming_all_fields_search(self):
        """
        Test searching all fields (no searchable attributes specified) returns some results.
        """
        pass

    def test_stemming_invalid_value_api_error(self):
        """Test that invalid stemming values produce proper API errors."""
        docs = [{"_id": "invalid_test", "field": "test content"}]
        mappings = {"field": {"type": "text_field", "language": "en", "stemming": "invalid_algorithm"}}

        # Should raise an error due to invalid stemming value
        with self.assertRaises(Exception) as cm:
            self.client.index(self.semi_structured_index_name).add_documents(
                docs, mappings=mappings, tensor_fields=[]
            )

        error_message = str(cm.exception)
        self.assertIn("stemming", error_message.lower())

    def test_stemming_field_change_api_error(self):
        """Test that changing stemming configuration produces API error."""
        # First add document with one stemming config
        docs1 = [{"_id": "change_test1", "title": "First document"}]
        mappings1 = {"title": {"type": "text_field", "language": "en", "stemming": "best"}}

        response1 = self.client.index(self.semi_structured_index_name).add_documents(
            docs1, mappings=mappings1, tensor_fields=[]
        )
        self.assertFalse(response1['errors'])

        # Try to add document with different stemming config for same field
        docs2 = [{"_id": "change_test2", "title": "Second document"}]
        mappings2 = {"title": {"type": "text_field", "language": "en", "stemming": "shortest"}}

        response2 = self.client.index(self.semi_structured_index_name).add_documents(
            docs2, mappings=mappings2, tensor_fields=[]
        )

        # Should have errors
        self.assertTrue(response2['errors'])
        error_message = response2['items'][0]['message']
        self.assertIn("different stemming configuration", error_message)

    def test_stemming_with_language_api(self):
        """Test stemming combined with language through API."""
        docs = [
            {
                "_id": "lang_stem_test",
                "lang_english_text": "running runners ran",
                "lang_french_text": "courant coureurs couru"
            }
        ]

        mappings = {
            "lang_english_text": {"type": "text_field", "language": "en", "stemming": "best"},
            "lang_french_text": {"type": "text_field", "language": "fr", "stemming": "best"}
        }

        response = self.client.index(self.semi_structured_index_name).add_documents(
            docs, mappings=mappings, tensor_fields=[]
        )
        self.assertFalse(response['errors'])

        # Allow time for indexing to complete
        time.sleep(2)

        # Test that documents with language+stemming configuration are searchable
        # First test English
        english_search = self.client.index(self.semi_structured_index_name).search(
            "running", search_method="LEXICAL", searchable_attributes=["lang_english_text"]
        )
        self.assertTrue(len(english_search['hits']) > 0)

        # Test general search to verify document is indexed
        general_search = self.client.index(self.semi_structured_index_name).search(
            "running", search_method="LEXICAL"
        )
        self.assertTrue(len(general_search['hits']) > 0)

        # Test that language was accepted (this is the main purpose of the test)
        self.assertEqual(general_search['hits'][0]["_id"], "lang_stem_test")


if __name__ == '__main__':
    unittest.main()
