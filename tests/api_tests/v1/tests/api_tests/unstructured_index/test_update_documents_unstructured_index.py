import uuid

from marqo.client import Client

from tests.marqo_test import MarqoTestCase


class TestUpdateDocumentsInUnstructuredIndex(MarqoTestCase):
    """
    Support for partial updates for unstructured indexes was added in 2.16.0.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.client = Client(**cls.client_settings)

        cls.text_index_name = "api_test_unstructured_index" + str(uuid.uuid4()).replace('-', '')

        cls.create_indexes([
            {
                "indexName": cls.text_index_name,
                "type": "unstructured",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "normalizeEmbeddings": False,
            }
        ])

        cls.indexes_to_delete = [cls.text_index_name]

    def tearDown(self):
        if self.indexes_to_delete:
            self.clear_indexes(self.indexes_to_delete)

    def test_update_document_with_ids(self):
        text_docs = [{
            '_id': '1',
            'tensor_field': 'title',
            'tensor_subfield': 'description',
            "short_string_field": "shortstring",
            "long_string_field": "Thisisaverylongstring" * 10,
            "int_field": 123,
            "float_field": 123.0,
            "string_array": ["aaa", "bbb"],
            "string_array2": ["123", "456"],
            "int_map": {"a": 1, "b": 2},
            "float_map": {"c": 1.0, "d": 2.0},
            "bool_field": True,
            "bool_field2": False,
            "custom_vector_field": {
                "content": "abcd",
                "vector": [1.0] * 32
            }
        }]

        mappings = {
            "custom_vector_field": {"type": "custom_vector"},
            "multimodal_combo_field": {
                "type": "multimodal_combination",
                "weights": {"tensor_field": 1.0, "tensor_subfield": 2.0}
            }
        }

        tensor_fields = ['tensor_field', 'custom_vector_field', 'multimodal_combo_field']

        add_docs_response = self.client.index(self.text_index_name).add_documents(documents = text_docs, mappings = mappings, tensor_fields = tensor_fields)

        assert add_docs_response["errors"] == False

        update_docs_response = self.client.index(self.text_index_name).update_documents(
            [{
                '_id': '1',
                'bool_field': False,
                'update_field_that_doesnt_exist': 500,
                'int_field': 1.0,
                'float_field': 500.0,
                'int_map': {
                    'a': 2,
                },
                'float_map': {
                    'c': 3.0,
                },
                'string_array': ["ccc"]
            }]
        )

        assert update_docs_response["errors"] == False

        get_docs_response = self.client.index(self.text_index_name).get_document(document_id = '1')

        assert get_docs_response['results'][0]['bool_field'] == False
        assert get_docs_response['results'][0]['int_field'] == 1.0
        assert get_docs_response['results'][0]['float_field'] == 500.0
        assert get_docs_response['results'][0]['int_map']['a'] == 2
        assert get_docs_response['results'][0]['float_map']['c'] == 3.0
        assert get_docs_response['results'][0]['string_array'] == ["ccc"]
        assert get_docs_response['results'][0]['update_field_that_doesnt_exist'] == 500
        assert get_docs_response['results'][0]['string_array2'] == ["123", "456"]

