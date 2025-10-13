"""This file contains two test suites:
    - TestFiltering which tests filtering on content with special characters
    - TestFieldnames which tests fieldnames with special characters.

TestFieldNames tests the following:
    - Identifies and tests which special characters can be indexed
    - Identifies and tests which special characters work in fieldnames used as searchable attributes
    - Identifies and tests which special characters work in fieldnames used as within filters
"""
import uuid

from marqo.errors import MarqoWebError

from tests.marqo_test import MarqoTestCase


class TestSpecialCharacterInValue(MarqoTestCase):
    """We test whether these special characters can be included in the
    value of a document for indexing and filtering"""
    supported_special_str_sequences = {
        '/', '*', '^', '\\', '!', '[', '||', '?',
        '&&', '"', ']', '-', '{', '~', '+', '}', ':', ')', '(', '.', '\n', '\t', '\r',
    }

    unsupported_special_str_sequences = {
        '\b', '\f', '\v', '\a', '\x08', '\x0c', '\x0b', '\x07'
    }

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.standard_structured_index_name = "structured_standard" + str(uuid.uuid4()).replace('-', '')
        cls.standard_unstructured_index_name = "unstructured_standard" + str(uuid.uuid4()).replace('-', '')

        cls.create_indexes([
            {
                "indexName": cls.standard_unstructured_index_name,
                "type": "unstructured"
            },
            {
                "indexName": cls.standard_structured_index_name,
                "type": "structured",
                "allFields": [{"name": "searchField", "type": "text", "features": ["lexical_search"]},
                              {"name": "filteringField", "type": "text", "features": ["filter"]}],
                "tensorFields": ["searchField"]
            }
        ])

        cls.indexes_to_delete = [cls.standard_unstructured_index_name, cls.standard_structured_index_name]

    def test_filtering_on_content_with_special_chars_tensor(self):
        for index_name in [self.standard_structured_index_name, self.standard_unstructured_index_name]:
            for special_str in self.supported_special_str_sequences:
                docs = [
                    {'_id': 'doc_0', 'filteringField': f"str_at_back_{special_str}", "searchField": "hello"},
                    {'_id': 'doc_1', 'filteringField': f"{special_str}_str_at_front", "searchField": "hello"},
                    {'_id': 'doc_2', 'filteringField': f"str_{special_str}_mid", "searchField": "hello"},
                    {'_id': 'doc_3', 'filteringField': f"{special_str}", "searchField": "hello"},
                ]
                for expected_document in docs:
                    str_for_filtering = expected_document['filteringField'].replace(special_str, f"\\{special_str}")

                    with self.subTest(f"Tensor Search, {index_name}, "
                                      f"str for filtering {str_for_filtering}, expected_doc {expected_document}"):
                        self.clear_indexes(self.indexes_to_delete)

                        self.client.index(index_name).add_documents(docs, tensor_fields=["searchField"] if \
                            index_name.startswith("un") else None)

                        res = self.client.index(index_name).search(q="hello",
                                                                   filter_string=f"filteringField:{str_for_filtering}", )

                        self.assertEqual(res["hits"][0]["_id"], expected_document["_id"])

    def test_filtering_on_content_with_special_chars_lexical(self):
        for index_name in [self.standard_structured_index_name, self.standard_unstructured_index_name]:
            for special_str in self.supported_special_str_sequences:
                docs = [
                    {'_id': 'doc_0', 'filteringField': f"str_at_back_{special_str}", "searchField": "hello"},
                    {'_id': 'doc_1', 'filteringField': f"{special_str}_str_at_front", "searchField": "hello"},
                    {'_id': 'doc_2', 'filteringField': f"str_{special_str}_mid", "searchField": "hello"},
                    {'_id': 'doc_3', 'filteringField': f"{special_str}", "searchField": "hello"},
                ]
                for expected_document in docs:
                    str_for_filtering = expected_document['filteringField'].replace(special_str, f"\\{special_str}")

                    with self.subTest(f"Lexical search, {index_name}, str for filtering "
                                      f"{str_for_filtering}, expected_doc {expected_document}"):
                        self.clear_indexes(self.indexes_to_delete)

                        self.client.index(index_name).add_documents(docs, tensor_fields=["searchField"] if \
                            index_name.startswith("un") else None)

                        res = self.client.index(index_name).search(q="hello",
                                                                   filter_string=f"filteringField:{str_for_filtering}",
                                                                   search_method="LEXICAL")

                        self.assertEqual(res["hits"][0]["_id"], expected_document["_id"])

    def test_unsupported_special_string_errors(self):
        bad_documents = [
            {'_id': f'doc_{i}', "searchField": f"{bad_string}"} for i, bad_string in \
            enumerate(self.unsupported_special_str_sequences)
        ]
        for index_name in [self.standard_structured_index_name, self.standard_unstructured_index_name]:
            with self.subTest(index_name):
                res = self.client.index(index_name).add_documents(bad_documents, tensor_fields=["searchField"] if \
                    index_name.startswith("un") else None)
                self.assertEqual(True, res["errors"])
                for item in res["items"]:
                    self.assertEqual(400, item["status"])
                    self.assertIn("The document contains invalid characters in the fields.", item["message"])


class TestSpecialCharsFieldNamesUnstructured(MarqoTestCase):
    """This test class tests the supported and illegal field names for structured and unstructured indexes."""
    supported_field_name = ['longdy3h0r', 'bkf3f1dvedkn', 'p7dqwtisyxpu', 'MyString', 'a', "id"]

    illegal_field_name = ['',  # empty string
                          '1starts_with_digit',
                          'contains space',
                          '$pecialchar',  # unsupported special char
                          '_tensor_facets',  # reserved field name
                          '_highlights',  # reserved field name
                          '_score',  # reserved field name
                          '_found'  # reserved field name
                          ]

    illegal_for_unstructured_index = ['::']

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.standard_structured_index_name = "structured_standard" + str(uuid.uuid4()).replace('-', '')
        cls.standard_unstructured_index_name = "unstructured_standard" + str(uuid.uuid4()).replace('-', '')

        cls.create_indexes([
            {
                "indexName": cls.standard_unstructured_index_name,
                "type": "unstructured"
            },
        ])
        cls.indexes_to_delete = [cls.standard_unstructured_index_name]

    def test_supported_field_name_in_fieldnames_unstructured(self):
        """Test supported special chars in field names can be indexed, searched, and filtered on"""
        index_name = self.standard_unstructured_index_name

        docs = [{"_id": f"special_{i}", f"field_{special_str}": "hello"} for i, special_str in enumerate(
            self.supported_field_name)]

        # Ensure these special chars can be used in field names
        tensor_fields = [f"field_{special_str}" for special_str in self.supported_field_name]
        res = self.client.index(index_name).add_documents(docs, tensor_fields=tensor_fields)
        self.assertEqual(res["errors"], False)

        for special_str in self.supported_field_name:
            with self.subTest(f"Supported field name: {repr(special_str)}"):
                escaped_field_name = f"field_{special_str}".replace(special_str, f"{special_str}")
                res = self.client.index(index_name).search(q="hello",
                                                           filter_string=f"{escaped_field_name}:hello")
                self.assertEqual(f"special_{self.supported_field_name.index(special_str)}",
                                 res["hits"][0]["_id"], )

    def test_illegal_field_name_unstructured(self):
        """Test illegal chars in field names for unstructured index"""
        index_name = self.standard_unstructured_index_name

        illegal_field_names = self.illegal_field_name + self.illegal_for_unstructured_index
        for illegal_field_name in illegal_field_names:
            with self.subTest(f"Illegal field name: {repr(illegal_field_name)}"):
                docs = [{f"{illegal_field_name}": "hello"}]

                # Ensure these special chars can be used in field names
                tensor_fields = [f"{illegal_field_name}"]
                res = self.client.index(index_name).add_documents(docs, tensor_fields=tensor_fields)

                self.assertEqual(res["errors"], True)
                for item in res["items"]:
                    self.assertEqual(400, item["status"])
                    self.assertIn("Field name", item["error"])

    def test_illegal_field_name_for_structured(self):
        for illegal_field_name in self.illegal_field_name:
            with self.subTest(f"Illegal field name: {repr(illegal_field_name)}"):
                allFields = [{"name": f"{illegal_field_name}", "type": "text"}]
                tensor_fields = [f"{illegal_field_name}"]
                with self.assertRaises(MarqoWebError) as e:
                    res = self.client.create_index(self.standard_structured_index_name, type="structured",
                                                   all_fields=allFields, tensor_fields=tensor_fields)

                self.assertIn("Field name", str(e.exception.message))
