import unittest

import pytest

from marqo.core.document.tensor_fields_container import TensorFieldsContainer
from marqo.core.exceptions import AddDocumentsError
from marqo.core.models.marqo_index import FieldType


@pytest.mark.unittest
class TestTensorFieldsContainer(unittest.TestCase):

    def setUp(self):
        self.container = TensorFieldsContainer(
            tensor_fields=['tensor_field1', 'tensor_field2', 'custom_vector_field1',
                           'custom_vector_field2', 'combo_field1', 'combo_field2'],
            custom_vector_fields=['custom_vector_field1', 'custom_vector_field2'],
            multimodal_combo_fields={
                'combo_field1': {'subfield1': 1.0},
                'combo_field2': {'subfield1': 2.0, 'tensor_field2': 5.0},
            }
        )

    def test_initialisation(self):
        self.assertTrue(self.container.is_custom_tensor_field('custom_vector_field1'))
        self.assertTrue(self.container.is_custom_tensor_field('custom_vector_field2'))
        self.assertFalse(self.container.is_custom_tensor_field('tensor_field1'))

        self.assertTrue(self.container.is_multimodal_field('combo_field1'))
        self.assertTrue(self.container.is_multimodal_field('combo_field2'))
        self.assertFalse(self.container.is_multimodal_field('combo_field3'))

        self.assertEquals({'subfield1', 'tensor_field2'}, self.container.get_multimodal_sub_fields())
        self.assertEquals({'subfield1': 1.0}, self.container.get_multimodal_field_mapping('combo_field1'))
        self.assertEquals({'subfield1': 2.0, 'tensor_field2': 5.0},
                          self.container.get_multimodal_field_mapping('combo_field2'))

        self.assertEquals(0, len(self.container._tensor_field_map))

    def test_collect_non_tensor_fields(self):
        test_cases = [
            (1, None),  # for unstructured, we don't infer type for non-text field
            (1, FieldType.Int),  # for structured, we pass in the type nevertheless
            (1.0, None),
            (1.0, FieldType.Float),
            (True, None),
            (True, FieldType.Bool),
            ('abcd', FieldType.Text),
            ('http://url', FieldType.ImagePointer),
            (['abcd', 'efg'], None),
            (['abcd', 'efg'], FieldType.ArrayText),
            ({'a': 1, 'b': 2}, None),
            ({'a': 1, 'b': 2}, FieldType.MapInt),
        ]
        for (field_content, field_type) in test_cases:
            with self.subTest(msg=f'field_content {field_content} of type {field_type}'):
                content = self.container.collect('doc_id1', 'field1', field_content, field_type)
                self.assertEquals(field_content, content)
                # verify that they won't be collected to tensor field maps
                self.assertEquals(0, len(self.container._tensor_field_map))

    def test_collect_custom_vector_field(self):
        content = self.container.collect('doc_id1', 'custom_vector_field1', {
            'content': 'content1',
            'vector': [1.0, 2.0]
        }, None)

        self.assertEquals('content1', content)
        self.assertIn('doc_id1', self.container._tensor_field_map)
        self.assertIn('custom_vector_field1', self.container._tensor_field_map['doc_id1'])

        tensor_field_content = self.container._tensor_field_map['doc_id1']['custom_vector_field1']
        self.assertEquals('content1', tensor_field_content.field_content)
        self.assertEquals(FieldType.CustomVector, tensor_field_content.field_type)
        self.assertEquals(['content1'], tensor_field_content.chunks)
        self.assertEquals([[1.0, 2.0]], tensor_field_content.embeddings)
        # TODO verify if this is also true, what if it's not included in the tensor_fields?
        self.assertTrue(tensor_field_content.is_tensor_field)
        self.assertFalse(tensor_field_content.is_multimodal_subfield)

    def test_collect_multimodal_field_should_raise_error(self):
        with self.assertRaises(AddDocumentsError) as e:
            self.container.collect('doc_id1', 'combo_field1', 'abc', FieldType.Text)

        self.assertIn("Field combo_field1 is a multimodal combination field and cannot be assigned a value.",
                      str(e.exception))

    def test_collect_tensor_field_with_non_string_type(self):
        test_cases = [
            (1, None),  # for unstructured, we don't infer type for non-text field
            (1, FieldType.Int),  # for structured, we pass in the type nevertheless
            (1.0, None),
            (1.0, FieldType.Float),
            (True, None),
            (True, FieldType.Bool),
            (['abcd', 'efg'], None),
            (['abcd', 'efg'], FieldType.ArrayText),
            ({'a': 1, 'b': 2}, None),
            ({'a': 1, 'b': 2}, FieldType.MapInt),
        ]

        for (field_content, field_type) in test_cases:
            with self.subTest(msg=f'field_content {field_content} of type {field_type}'):
                with self.assertRaises(AddDocumentsError) as e:
                    self.container.collect('doc_id1', 'tensor_field1', field_content, field_type)

                self.assertIn(f"Invalid type {type(field_content)} for tensor field tensor_field1",
                              str(e.exception))

    def test_collect_tensor_field_with_string_type(self):
        for text_field_type in [FieldType.Text, FieldType.ImagePointer, FieldType.AudioPointer, FieldType.VideoPointer]:
            with self.subTest(msg=f'field_type {text_field_type}'):
                content = self.container.collect('doc_id1', 'tensor_field1', 'content', text_field_type)
                self.assertEquals('content', content)
                self.assertIn('doc_id1', self.container._tensor_field_map)
                self.assertIn('tensor_field1', self.container._tensor_field_map['doc_id1'])

                tensor_field_content = self.container._tensor_field_map['doc_id1']['tensor_field1']
                self.assertEquals('content', tensor_field_content.field_content)
                self.assertEquals(text_field_type, tensor_field_content.field_type)
                self.assertEquals([], tensor_field_content.chunks)
                self.assertEquals([], tensor_field_content.embeddings)
                self.assertTrue(tensor_field_content.is_tensor_field)
                self.assertFalse(tensor_field_content.is_multimodal_subfield)

    def test_collect_tensor_field_can_identify_toplevel_or_subfield(self):
        test_cases = [
            ('tensor_field1', True, False),
            ('tensor_field2', True, True),
            ('subfield1', False, True),
        ]

        for (field_name, is_tensor_field, is_multimodal_subfield) in test_cases:
            with self.subTest(msg=f'{field_name}: is_tensor_field={is_tensor_field}, is_multimodal_subfield={is_multimodal_subfield}'):
                self.container.collect('doc_id1', field_name, 'content', FieldType.Text)

                tensor_field_content = self.container._tensor_field_map['doc_id1'][field_name]
                self.assertEquals(is_tensor_field, tensor_field_content.is_tensor_field)
                self.assertEquals(is_multimodal_subfield, tensor_field_content.is_multimodal_subfield)

    def test_remove_doc(self):
        self.container.collect('doc_id1', 'tensor_field1', 'content', FieldType.Text)
        self.container.collect('doc_id1', 'tensor_field2', 'content', FieldType.Text)
        self.container.collect('doc_id2', 'tensor_field2', 'content', FieldType.Text)

        self.assertIn('doc_id1', self.container._tensor_field_map)
        self.assertIn('doc_id2', self.container._tensor_field_map)
        self.assertNotIn('doc_id3', self.container._tensor_field_map)

        self.container.remove_doc('doc_id1')
        self.container.remove_doc('doc_id3')

        self.assertNotIn('doc_id1', self.container._tensor_field_map)
        self.assertIn('doc_id2', self.container._tensor_field_map)
