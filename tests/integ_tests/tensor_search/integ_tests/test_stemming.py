import unittest
import uuid

from marqo.core.models.marqo_index import *
from marqo.core.models.marqo_index_request import UnstructuredMarqoIndexRequest
from marqo.core.models.hybrid_parameters import HybridParameters, RetrievalMethod, RankingMethod
from marqo.tensor_search import tensor_search
from marqo.core.models.add_docs_params import AddDocsParams
from tests.integ_tests.marqo_test import MarqoTestCase


class TestStemmingIntegration(MarqoTestCase):
    """
    Integration tests for text field stemming functionality.
    """

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # Create semi-structured index for stemming tests
        cls.stemming_index = cls.unstructured_marqo_index_request(
            name="stemming-integ-" + str(uuid.uuid4()).replace('-', ''),
            model=Model(name="hf/e5-small-v2")
        )

        # 2.16 unstructured index -- must support stemming
        cls.v216_unstructured_index = cls.unstructured_marqo_index_request(
            name='test_v216_stemming_' + str(uuid.uuid4()).replace('-', ''),
            model=Model(name='hf/e5-small-v2'),
            marqo_version='2.16.0',
        )

        # 2.15 unstructured index -- must not support stemming
        cls.v215_unstructured_index = cls.unstructured_marqo_index_request(
            name='test_v215_stemming_' + str(uuid.uuid4()).replace('-', ''),
            model=Model(name='hf/e5-small-v2'),
            marqo_version='2.15.0',
        )

        index_requests = [cls.stemming_index, cls.v216_unstructured_index, cls.v215_unstructured_index]
        cls.indexes = cls.create_indexes(index_requests)

    def populate_index(self):
        """Populate index with stemming test documents."""
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

        mappings = {
            "title_stem1": {"type": "text_field", "language": "de", "stemming": "none"},
            "title_stem2": {"type": "text_field", "language": "de", "stemming": "best"},
            "title_stem3": {"type": "text_field", "language": "de", "stemming": "shortest"},
            "title_stem4": {"type": "text_field", "language": "de", "stemming": "multiple"},
        }

        add_docs_params = AddDocsParams(
            index_name=self.stemming_index.name,
            docs=docs,
            mappings=mappings,
            tensor_fields=["title_stem1"]
        )

        res = self.add_documents(
            config=self.config,
            add_docs_params=add_docs_params
        )

        self.assertFalse(res.errors, "Should not have errors when adding documents")

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
                "nacionalmente", ["title_stem1", "title_stem2"], ["1", "2"],
                "Full word matches with none and best fields"
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
            ("LEXICAL", None, "lexical search"),
            ("HYBRID", HybridParameters(retrievalMethod=RetrievalMethod.Lexical, rankingMethod=RankingMethod.Lexical),
             "hybrid lexical/lexical"),
            ("HYBRID", HybridParameters(alpha=0), "hybrid RRF with alpha=0")
        ]

        self.populate_index()

        for query, fields, expected_ids, description in cases:
            for search_method, hybrid_params, search_description in search_configs:
                with self.subTest(f"{search_method} search for '{query}' in {fields}: {description}"):
                    if search_method == "LEXICAL":
                        res = tensor_search.search(
                            config=self.config,
                            index_name=self.stemming_index.name,
                            text=query,
                            search_method=search_method,
                            searchable_attributes=fields,
                            result_count=10,
                            offset=0,
                            language="de"
                        )
                    else:  # HYBRID
                        # For hybrid search, need to specify fields in hybrid parameters
                        hybrid_params_with_fields = HybridParameters(
                            retrievalMethod=hybrid_params.retrievalMethod,
                            rankingMethod=hybrid_params.rankingMethod,
                            alpha=hybrid_params.alpha,
                            searchableAttributesLexical=fields
                        )
                        res = tensor_search.search(
                            config=self.config,
                            index_name=self.stemming_index.name,
                            text=query,
                            search_method=search_method,
                            result_count=10,
                            offset=0,
                            language="de",
                            hybrid_parameters=hybrid_params_with_fields
                        )

                    actual_ids = set(hit["_id"] for hit in res["hits"] if hit["_id"] in expected_ids)
                    self.assertEqual(set(expected_ids), actual_ids, f"Failed for query '{query}' in fields {fields}")

    def test_stemming_all_fields_search(self):
        """
        Test searching all fields (no searchable attributes specified) returns some results.
        """
        self.populate_index()

        res = tensor_search.search(
            config=self.config,
            index_name=self.stemming_index.name,
            text="nacionalmente",
            search_method="LEXICAL",
            result_count=10,
            offset=0,
            language="de"
        )

        self.assertGreater(len(res["hits"]), 0, "Should find matches for 'nacionalmente' in all fields")

    def test_stemming_invalid_value_error(self):
        """Test that invalid stemming values produce proper errors."""
        docs = [{"_id": "invalid_test", "field": "test content"}]
        mappings = {"field": {"type": "text_field", "language": "en", "stemming": "invalid_algorithm"}}

        add_docs_params = AddDocsParams(
            index_name=self.stemming_index.name,
            docs=docs,
            mappings=mappings,
            tensor_fields=[]
        )

        # Should raise an error due to invalid stemming value
        with self.assertRaises(Exception) as cm:
            self.add_documents(
                config=self.config,
                add_docs_params=add_docs_params
            )

        error_message = str(cm.exception)
        self.assertIn("stemming", error_message.lower())

    def test_stemming_field_change_error(self):
        """Test that changing stemming configuration produces error."""
        # First add document with one stemming config
        docs1 = [{"_id": "change_test1", "title": "First document"}]
        mappings1 = {"title": {"type": "text_field", "language": "en", "stemming": "best"}}

        add_docs_params1 = AddDocsParams(
            index_name=self.stemming_index.name,
            docs=docs1,
            mappings=mappings1,
            tensor_fields=[]
        )

        response1 = self.add_documents(
            config=self.config,
            add_docs_params=add_docs_params1
        )
        self.assertFalse(response1.errors)

        # Try to add document with different stemming config for same field
        docs2 = [{"_id": "change_test2", "title": "Second document"}]
        mappings2 = {"title": {"type": "text_field", "language": "en", "stemming": "shortest"}}

        add_docs_params2 = AddDocsParams(
            index_name=self.stemming_index.name,
            docs=docs2,
            mappings=mappings2,
            tensor_fields=[]
        )

        response2 = self.add_documents(
            config=self.config,
            add_docs_params=add_docs_params2
        )

        # Should have errors
        self.assertTrue(response2.errors)
        error_message = response2.items[0].message
        self.assertIn("different stemming configuration", error_message)

    def test_stemming_no_language(self):
        """Test that no stemming occurs when stemming is set to 'none' without language in field mapping."""
        docs = [
            {
                "_id": "no_lang_1",
                "content": "apples"
            },
            {
                "_id": "no_lang_2",
                "content": "apple"
            }
        ]

        # Add documents with no stemming and no language in mapping
        mappings = {
            "content": {"type": "text_field", "stemming": "none"}
        }

        add_docs_params = AddDocsParams(
            index_name=self.stemming_index.name,
            docs=docs,
            mappings=mappings,
            tensor_fields=[]
        )

        res = self.add_documents(
            config=self.config,
            add_docs_params=add_docs_params
        )

        self.assertFalse(res.errors, "Should not have errors when adding documents without language")

        # Search for exact matches should work
        apples_res = tensor_search.search(
            config=self.config,
            index_name=self.stemming_index.name,
            text="apples",
            search_method="LEXICAL",
            searchable_attributes=["content"],
            result_count=10,
            offset=0,
            language="en"
        )

        apple_res = tensor_search.search(
            config=self.config,
            index_name=self.stemming_index.name,
            text="apple",
            search_method="LEXICAL",
            searchable_attributes=["content"],
            result_count=10,
            offset=0,
            language="en"
        )

        # Verify exact matches work
        apples_ids = {hit["_id"] for hit in apples_res["hits"]}
        apple_ids = {hit["_id"] for hit in apple_res["hits"]}

        self.assertEqual({"no_lang_1"}, apples_ids, "Should find document with 'running'")
        self.assertEqual({"no_lang_2"}, apple_ids, "Should find document with 'runs'")

    def test_stemming_not_specified(self):
        """Test that stemming configuration persists when not specified in subsequent document additions."""
        # First add document with explicit stemming=none
        docs1 = [{"_id": "doc1", "text_field": "green apples"}]
        mappings1 = {"text_field": {"type": "text_field", "language": "en", "stemming": "none"}}

        add_docs_params1 = AddDocsParams(
            index_name=self.stemming_index.name,
            docs=docs1,
            mappings=mappings1,
            tensor_fields=[]
        )

        response1 = self.add_documents(
            config=self.config,
            add_docs_params=add_docs_params1
        )
        self.assertFalse(response1.errors, "Should not have errors when adding first document")

        # Add second document to same field without specifying stemming
        docs2 = [{"_id": "doc2", "text_field": "green apple"}]
        mappings2 = {"text_field": {"type": "text_field", "language": "en"}}  # No stemming specified

        add_docs_params2 = AddDocsParams(
            index_name=self.stemming_index.name,
            docs=docs2,
            mappings=mappings2,
            tensor_fields=[]
        )

        response2 = self.add_documents(
            config=self.config,
            add_docs_params=add_docs_params2
        )
        self.assertFalse(response2.errors, "Should not have errors when adding second document")

        # Verify exact matches still work as expected
        apples_res = tensor_search.search(
            config=self.config,
            index_name=self.stemming_index.name,
            text="apples",
            search_method="LEXICAL",
            searchable_attributes=["text_field"],
            result_count=10,
            offset=0,
            language="en"
        )

        apple_res = tensor_search.search(
            config=self.config,
            index_name=self.stemming_index.name,
            text="apple",
            search_method="LEXICAL",
            searchable_attributes=["text_field"],
            result_count=10,
            offset=0,
            language="en"
        )

        apples_ids = {hit["_id"] for hit in apples_res["hits"]}
        apple_ids = {hit["_id"] for hit in apple_res["hits"]}

        self.assertEqual({"doc1"}, apples_ids, "Should find first document with exact match 'running'")
        self.assertEqual({"doc2"}, apple_ids, "Should find second document with exact match 'runs'")

    def test_vespa_schema_contains_language_and_stemming(self):
        """
        Test that the generated Vespa schema contains both language and stemming configuration when both are specified.
        """
        # Add document with both language and stemming specified
        docs = [{"_id": "schema_test", "schema_field": "testing words"}]
        mappings = {"schema_field": {"type": "text_field", "language": "en", "stemming": "best"}}

        add_docs_params = AddDocsParams(
            index_name=self.stemming_index.name,
            docs=docs,
            mappings=mappings,
            tensor_fields=[]
        )

        response = self.add_documents(
            config=self.config,
            add_docs_params=add_docs_params
        )
        self.assertFalse(response.errors, "Should not have errors when adding document")

        # Get the actual deployed Vespa schema content
        # First get the MarqoIndex to find the schema name
        marqo_index = self.config.vespa_client.get_index_setting_by_name(self.stemming_index.name)
        self.assertIsNotNone(marqo_index, "Should be able to retrieve the index settings")

        content_base_url = f"{self.config.vespa_client.config_url}/application/v2/tenant/default/application/default/environment/prod/region/default/instance/default/content"
        schema_file_name = f"{marqo_index.schema_name}.sd"

        try:
            schema_content = self.config.vespa_client.get_text_content(content_base_url, "/schemas/", schema_file_name)
        except Exception as e:
            self.fail(f"Could not retrieve schema content: {e}")

        # Parse the schema content to find the field definition
        schema_field_found = False
        field_lines = []
        in_field_block = False
        brace_count = 0

        for line in schema_content.split('\n'):
            if f'field marqo__lexical_schema_field' in line and 'type string' in line:
                schema_field_found = True
                in_field_block = True
                field_lines.append(line.strip())
                brace_count += line.count('{') - line.count('}')
            elif in_field_block:
                field_lines.append(line.strip())
                brace_count += line.count('{') - line.count('}')
                if brace_count == 0:  # End of field block
                    break

        self.assertTrue(schema_field_found, "Should find the schema_field in the generated Vespa schema")

        # Check that the field definition contains both language and stemming
        field_definition = '\n'.join(field_lines)

        # Language is set using set_language in Vespa indexing pipeline
        self.assertIn('"en" | set_language', field_definition,
                      "Field should have language set to 'en' via set_language in the Vespa schema")

        # Stemming is set as a field property in Vespa schema
        self.assertIn('stemming: best', field_definition,
                      "Field should have stemming set to 'best' in the Vespa schema")

    def test_v216_unstructured_index_supports_stemming_mapping(self):
        """Test that v2.16 unstructured index supports stemming mapping."""
        docs = [
            {"_id": "v216_doc1", "title": "nacionalmente", "content": "German word example"},
            {"_id": "v216_doc2", "title": "andere Wörter", "content": "Other German words"},
            {"_id": "v216_doc3", "title": "mehr Text", "content": "More German text"}
        ]

        mappings = {
            "title": {"type": "text_field", "language": "de", "stemming": "best"},
            "content": {"type": "text_field", "language": "de", "stemming": "shortest"}
        }

        # Add documents with stemming mapping
        response = self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.v216_unstructured_index.name,
                docs=docs,
                tensor_fields=["title"],
                mappings=mappings
            )
        )

        # Verify documents were added successfully
        self.assertFalse(response.errors, "Failed to add documents with stemming mapping to v2.16 index")

        # Test search with stemmed word - should find matches due to stemming  
        # "nacionalment" should match "nacionalmente" with German stemming
        result = tensor_search.search(
            config=self.config,
            index_name=self.v216_unstructured_index.name,
            text="nacionalment",
            search_method="LEXICAL",
            searchable_attributes=["title"],
            language="de"
        )

        # Verify we get hits (stemmed "nacionalment" should match "nacionalmente")
        self.assertGreater(len(result["hits"]), 0, "Expected hits for stemmed search on v2.16 index")
        hit_ids = [hit["_id"] for hit in result["hits"]]
        self.assertIn("v216_doc1", hit_ids, "Expected v216_doc1 in search results for stemmed query")

    def test_v215_unstructured_index_rejects_stemming_mapping(self):
        """Test that v2.15 unstructured index rejects stemming mapping."""
        docs = [
            {"_id": "v215_doc1", "title": "nacionalmente", "content": "example word"},
            {"_id": "v215_doc2", "title": "andere", "content": "other words"}
        ]

        mappings = {
            "title": {"type": "text_field", "stemming": "best"},
            "content": {"type": "text_field", "stemming": "shortest"}
        }

        # Try to add documents with stemming mapping - this should fail
        response = self.add_documents(
            config=self.config,
            add_docs_params=AddDocsParams(
                index_name=self.v215_unstructured_index.name,
                docs=docs,
                tensor_fields=["title"],
                mappings=mappings
            )
        )

        # Verify that adding documents with stemming mapping fails
        self.assertTrue(
            response.errors,
            "Expected error when adding documents with stemming mapping to v2.15 index"
        )

        # Check the specific error message content in the response items
        error_items = [item for item in response.items if item.error]
        self.assertEqual(2, len(error_items), "Expected two error items in response")

        # Verify all items have the stemming version error
        for error_item in error_items:
            self.assertIn(
                "Stemming is only supported for indexes created with Marqo version 2.16.0 or later",
                error_item.error
            )
            self.assertIn("2.15.0", error_item.error)


if __name__ == '__main__':
    unittest.main()
