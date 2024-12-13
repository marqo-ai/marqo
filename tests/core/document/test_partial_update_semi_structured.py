from typing import List, Dict, Any

from marqo.core.models.add_docs_params import AddDocsParams
from marqo.tensor_search import tensor_search
from tests.marqo_test import MarqoTestCase


class TestPartialUpdate(MarqoTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # semi_structured_index_request = cls.unstructured_marqo_index_request(name='test_partial_update_semi_structured')
        # cls.create_indexes([semi_structured_index_request])
        cls.index = cls.config.index_management.get_index('test_partial_update_semi_structured')

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()
        self.doc = {
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
        }
        self.add_documents(self.config, add_docs_params=AddDocsParams(
            index_name=self.index.name,
            docs=[self.doc],
            tensor_fields=['title', 'custom_vector_field', 'multimodal_combo_field'],
            mappings={
                "custom_vector_field": {"type": "custom_vector"},
                "multimodal_combo_field": {
                    "type": "multimodal_combination",
                    "weights": {"tensor_field": 1.0, "tensor_subfield": 2.0}
                }
            }
        ))

    def _assert_fields_unchanged(self, doc: Dict[str, Any], excluded_fields: List[str]):
        for field, value in self.doc.items():
            if field in excluded_fields:
                continue
            elif field == 'custom_vector_field':
                self.assertEqual(value['content'], doc.get(field, None), f'{field} is changed.')
            elif isinstance(value, dict):
                for k, v in value.items():
                    flattened_field_name = f'{field}.{k}'
                    self.assertEqual(v, doc.get(flattened_field_name, None), f'{flattened_field_name} is changed.')
            else:
                self.assertEqual(value, doc.get(field, None), f'{field} is changed.')

    # Test update single field
    def test_partial_update_should_update_bool_field(self):
        res = self.config.document.partial_update_documents([{'_id': '1', 'bool_field': False}], self.index)
        self.assertFalse(res.errors)

        doc = tensor_search.get_document_by_id(self.config, self.index.name, '1')
        self.assertFalse(doc['bool_field'])
        self._assert_fields_unchanged(doc, ['bool_field'])

    def test_partial_update_should_update_int_field_to_int(self):
        res = self.config.document.partial_update_documents([{'_id': '1', 'int_field': 500}], self.index)
        self.assertFalse(res.errors)

        doc = tensor_search.get_document_by_id(self.config, self.index.name, '1')
        self.assertEqual(500, doc['int_field'])
        self._assert_fields_unchanged(doc, ['int_field'])

    def test_partial_update_should_update_int_field_to_float(self):
        res = self.config.document.partial_update_documents([{'_id': '1', 'int_field': 1.0}], self.index)
        self.assertFalse(res.errors)

        doc = tensor_search.get_document_by_id(self.config, self.index.name, '1')
        self.assertEqual(1.0, doc['int_field'])
        self._assert_fields_unchanged(doc, ['int_field'])

    def test_partial_update_should_update_float_field_to_float(self):
        pass

    def test_partial_update_should_update_int_map(self):
        pass

    def test_partial_update_should_update_float_map(self):
        pass

    def test_partial_update_should_allow_changing_numeric_types_in_map(self):
        res = self.config.document.partial_update_documents([{'_id': '1', 'int_map': {
            'a': 2.0,  # update int to float
            'c': 3,  # add new int value
            'd': 4.0  # add new float value
        }}], self.index)
        self.assertFalse(res.errors)

        doc = tensor_search.get_document_by_id(self.config, self.index.name, '1')
        self.assertEqual(2.0, doc['int_map.a'])
        self.assertNotIn('int_map.b', doc)  # b will be deleted
        self.assertEqual(3, doc['int_map.c'])
        self.assertEqual(4.0, doc['int_map.d'])
        self._assert_fields_unchanged(doc, ['int_map'])

    def test_partial_update_should_update_string_array(self):
        res = self.config.document.partial_update_documents([{'_id': '1', 'string_array': ["ccc"]}], self.index)
        self.assertFalse(res.errors)

        doc = tensor_search.get_document_by_id(self.config, self.index.name, '1')
        self.assertEqual(["ccc"], doc['string_array'])
        self._assert_fields_unchanged(doc, ['string_array'])

    def test_partial_update_should_update_short_string(self):
        pass

    def test_partial_update_should_update_long_string(self):
        pass

    def test_partial_update_should_update_long_string_to_short_string(self):
        res = tensor_search.search(self.config, self.index.name, text='*',
                                   filter=f'long_string_field:{self.doc["long_string_field"]}')
        self.assertEqual(0, len(res['hits']))

        res = self.config.document.partial_update_documents([{'_id': '1', 'long_string_field': 'short'}], self.index)
        print(res)
        self.assertFalse(res.errors)

        doc = tensor_search.get_document_by_id(self.config, self.index.name, '1')
        self.assertEqual('short', doc['long_string_field'])
        self._assert_fields_unchanged(doc, ['long_string_field'])

        res = tensor_search.search(self.config, self.index.name, text='*', filter=f'long_string_field:short')
        self.assertEqual(1, len(res['hits']))

    def test_partial_update_should_update_short_string_to_long_string(self):
        res = tensor_search.search(self.config, self.index.name, text='*',
                                   filter=f'short_string_field:{self.doc["short_string_field"]}')
        self.assertEqual(1, len(res['hits']))

        res = self.config.document.partial_update_documents([{'_id': '1', 'short_string_field': 'verylongstring'*10}], self.index)
        print(res)
        self.assertFalse(res.errors)

        doc = tensor_search.get_document_by_id(self.config, self.index.name, '1')
        self.assertEqual('verylongstring'*10, doc['short_string_field'])
        self._assert_fields_unchanged(doc, ['short_string_field'])

        res = tensor_search.search(self.config, self.index.name, text='*',
                                   filter=f'short_string_field:{doc["short_string_field"]}')
        self.assertEqual(0, len(res['hits']))

    def test_partial_update_should_update_score_modifiers(self):
        pass

    # Test update multiple fields
    def test_partial_update_should_update_multiple_fields(self):
        pass

    # Test remove field
    def test_partial_update_should_remove_field_if_set_to_none(self):
        pass

    # Test add new fields
    def test_partial_update_should_add_new_fields(self):
        pass

    # Reject any tensor field change
    def test_partial_update_should_reject_tensor_field(self):
        pass

    def test_partial_update_should_reject_tensor_subfield(self):
        pass

    def test_partial_update_should_reject_custom_vector_field(self):
        pass

    def test_partial_update_should_reject_multimodal_combo_field(self):
        pass

    # Other edge cases
    # * reject numeric array field type
    # * reject adding new lexical field
    # * invalid field name
    # * invalid contents

    # Concurrent update, last write wins
    # Concurrent update with add doc override, last write winds

    # Perf test cases
    # 100 docs with each: 2 tensor fields, 1 multimodal, 10 text field, 10 int field, 10 float field, 10 bool field,
    # 2 int map with 10 items, 2 float map with 10 items, 2 string array fields with 10 items each

    # Update 100 docs 10 times
    # * update only 1 text field
    # * update only 1 bool field
    # * update only 1 int field
    # * update only 1 int map
    # * update only 1 float field
    # * update only 1 float map
    # * update only 1 string array

    # Add doc override 100 docs 10 times, with using existing tensor enabled
