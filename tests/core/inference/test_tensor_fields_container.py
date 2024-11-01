import unittest
from typing import cast

import pytest

from marqo.core.constants import MARQO_DOC_ID, MARQO_DOC_TENSORS, MARQO_DOC_CHUNKS, MARQO_DOC_EMBEDDINGS
from marqo.core.inference.tensor_fields_container import TensorFieldsContainer, MultiModalTensorFieldContent
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
            },
            should_normalise_custom_vector=True
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
        self.assertEquals([[0.4472135954999579, 0.8944271909999159]], tensor_field_content.embeddings)  # normalised
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

    def test_collect_multimodal_fields_should_return_all(self):
        fields = list(self.container.collect_multi_modal_fields('doc_id1', True))
        self.assertEquals(('combo_field1', {'subfield1': 1.0}), fields[0])
        self.assertEquals(('combo_field2', {'subfield1': 2.0, 'tensor_field2': 5.0}), fields[1])

    def test_collect_multimodal_fields_should_populate_subfields(self):
        self.container.collect('doc_id1', 'tensor_field2', 'tensor_field2_content', FieldType.Text)
        self.container.collect('doc_id1', 'subfield1', 'subfield1_content', FieldType.Text)

        list(self.container.collect_multi_modal_fields('doc_id1', True))

        self.assertIn('doc_id1', self.container._tensor_field_map)
        self.assertIn('combo_field1', self.container._tensor_field_map['doc_id1'])

        combo_field1 = cast(MultiModalTensorFieldContent, self.container._tensor_field_map['doc_id1']['combo_field1'])
        self.assertEquals(FieldType.MultimodalCombination, combo_field1.field_type)
        self.assertEquals('', combo_field1.field_content)
        self.assertTrue(combo_field1.is_tensor_field)
        self.assertFalse(combo_field1.is_multimodal_subfield)
        self.assertEquals({'subfield1': 1.0}, combo_field1.weights)
        self.assertEquals({'subfield1': self.container._tensor_field_map['doc_id1']['subfield1']},
                          combo_field1.subfields)
        self.assertTrue(combo_field1.normalize_embeddings)

        combo_field2 = cast(MultiModalTensorFieldContent, self.container._tensor_field_map['doc_id1']['combo_field2'])
        self.assertEquals({'subfield1': self.container._tensor_field_map['doc_id1']['subfield1'],
                           'tensor_field2': self.container._tensor_field_map['doc_id1']['tensor_field2']},
                          combo_field2.subfields)

    def test_collect_multimodal_fields_should_not_populate_subfields_not_existing(self):
        self.container.collect('doc_id1', 'tensor_field2', 'tensor_field2_content', FieldType.Text)

        list(self.container.collect_multi_modal_fields('doc_id1', True))

        combo_field1 = cast(MultiModalTensorFieldContent, self.container._tensor_field_map['doc_id1']['combo_field1'])
        self.assertEquals({}, combo_field1.subfields)

        combo_field2 = cast(MultiModalTensorFieldContent, self.container._tensor_field_map['doc_id1']['combo_field2'])
        self.assertEquals({'tensor_field2': self.container._tensor_field_map['doc_id1']['tensor_field2']},
                          combo_field2.subfields)

    def test_populate_tensor_from_existing_docs_will_not_populate_if_doc_id_does_not_match(self):
        self.container.collect('doc_id1', 'tensor_field1', 'tensor_field1_content', FieldType.Text)
        tensor_field1 = self.container._tensor_field_map['doc_id1']['tensor_field1']

        self.container.populate_tensor_from_existing_doc({
            MARQO_DOC_ID: 'doc_id2',
            'tensor_field1': 'tensor_field1_content',
            MARQO_DOC_TENSORS: {
                'tensor_field1': {MARQO_DOC_CHUNKS: ['tensor_field1_content'], MARQO_DOC_EMBEDDINGS: [[1.0, 2.0]]}
            }
        }, {})

        self.assertEquals([], tensor_field1.chunks)
        self.assertEquals([], tensor_field1.embeddings)

    def test_populate_tensor_from_existing_docs_should_populate_if_doc_id_matches(self):
        self.container.collect('doc_id1', 'tensor_field1', 'tensor_field1_content', FieldType.Text)
        tensor_field1 = self.container._tensor_field_map['doc_id1']['tensor_field1']

        self.container.populate_tensor_from_existing_doc({
            MARQO_DOC_ID: 'doc_id1',
            'tensor_field1': 'tensor_field1_content',
            MARQO_DOC_TENSORS: {
                'tensor_field1': {MARQO_DOC_CHUNKS: ['tensor_field1_content'], MARQO_DOC_EMBEDDINGS: [[1.0, 2.0]]}
            }
        }, {})

        self.assertEquals(['tensor_field1_content'], tensor_field1.chunks)
        self.assertEquals([[1.0, 2.0]], tensor_field1.embeddings)

    def test_populate_tensor_from_existing_docs_will_not_populate_if_content_changes(self):
        self.container.collect('doc_id1', 'tensor_field1', 'changed_content', FieldType.Text)
        tensor_field1 = self.container._tensor_field_map['doc_id1']['tensor_field1']

        self.container.populate_tensor_from_existing_doc({
            MARQO_DOC_ID: 'doc_id1',
            'tensor_field1': 'tensor_field1_content',
            MARQO_DOC_TENSORS: {
                'tensor_field1': {MARQO_DOC_CHUNKS: ['tensor_field1_content'], MARQO_DOC_EMBEDDINGS: [[1.0, 2.0]]}
            }
        }, {})

        self.assertEquals([], tensor_field1.chunks)
        self.assertEquals([], tensor_field1.embeddings)

    def test_populate_tensor_from_existing_docs_will_not_populate_if_field_does_not_exist(self):
        self.container.collect('doc_id1', 'tensor_field1', 'tensor_field1_content', FieldType.Text)
        tensor_field1 = self.container._tensor_field_map['doc_id1']['tensor_field1']

        self.container.populate_tensor_from_existing_doc({
            MARQO_DOC_ID: 'doc_id1',
            'tensor_field2': 'tensor_field2_content',  # tensor_field1 does not exist in the existing doc
            MARQO_DOC_TENSORS: {}
        }, {})

        self.assertEquals([], tensor_field1.chunks)
        self.assertEquals([], tensor_field1.embeddings)

    def test_populate_tensor_from_existing_docs_will_not_populate_if_embedding_does_not_exist(self):
        self.container.collect('doc_id1', 'tensor_field1', 'tensor_field1_content', FieldType.Text)
        tensor_field1 = self.container._tensor_field_map['doc_id1']['tensor_field1']

        self.container.populate_tensor_from_existing_doc({
            MARQO_DOC_ID: 'doc_id1',
            'tensor_field1': 'tensor_field1_content',
            MARQO_DOC_TENSORS: {}  # embedding for tensor_field1 does not exist in the existing doc
        }, {})

        self.assertEquals([], tensor_field1.chunks)
        self.assertEquals([], tensor_field1.embeddings)

    def test_populate_tensor_from_existing_docs_will_not_populate_if_existing_field_is_multimodal_combo_field(self):
        self.container.collect('doc_id1', 'tensor_field1', 'tensor_field1_content', FieldType.Text)
        tensor_field1 = self.container._tensor_field_map['doc_id1']['tensor_field1']

        self.container.populate_tensor_from_existing_doc({
            MARQO_DOC_ID: 'doc_id1',
            MARQO_DOC_TENSORS: {
                'tensor_field1': {MARQO_DOC_CHUNKS: ['tensor_field1_content'], MARQO_DOC_EMBEDDINGS: [[1.0, 2.0]]}
            }
        }, {'tensor_field1': {'subfield1': 1.0}})  # tensor_field1 is a multimodal combo field

        self.assertEquals([], tensor_field1.chunks)
        self.assertEquals([], tensor_field1.embeddings)

    def test_populate_tensor_from_existing_docs_will_not_populate_for_custom_vector_field(self):
        self.container.collect('doc_id1', 'custom_vector_field1', {
            'content': 'content1',
            'vector': [1.0, 2.0]
        }, None)
        custom_vector_field1 = self.container._tensor_field_map['doc_id1']['custom_vector_field1']

        self.container.populate_tensor_from_existing_doc({
            MARQO_DOC_ID: 'doc_id1',
            'custom_vector_field1': 'content1',
            MARQO_DOC_TENSORS: {
                'custom_vector_field1': {MARQO_DOC_CHUNKS: ['content2'], MARQO_DOC_EMBEDDINGS: [[3.0, 4.0]]}
            }  # embedding for tensor_field1 does not exist in the existing doc
        }, {})

        self.assertEquals(['content1'], custom_vector_field1.chunks)
        self.assertEquals([[0.4472135954999579, 0.8944271909999159]], custom_vector_field1.embeddings)

    def test_populate_tensor_from_existing_docs_will_not_populate_for_multimodal_field_if_it_does_not_exist(self):
        combo_field2 = self._get_combo_field2()

        self.container.populate_tensor_from_existing_doc({
            MARQO_DOC_ID: 'doc_id1',
            'tensor_field2': 'tensor_field2_content',
            'subfield1': 'subfield1_content',
            MARQO_DOC_TENSORS: {}  # embedding for combo_field2 does not exist in the existing doc
        }, {})

        self.assertEquals([], combo_field2.chunks)
        self.assertEquals([], combo_field2.embeddings)

    def test_populate_tensor_from_existing_docs_will_not_populate_multimodal_field_with_another_type(self):
        combo_field2 = self._get_combo_field2()

        self.container.populate_tensor_from_existing_doc({
            MARQO_DOC_ID: 'doc_id1',
            'tensor_field2': 'tensor_field2_content',
            'subfield1': 'subfield1_content',
            'combo_field2': 'combo_field2_content',
            MARQO_DOC_TENSORS: {
                'combo_field2': {MARQO_DOC_CHUNKS: ['combo_field2_content'], MARQO_DOC_EMBEDDINGS: [[1.0, 2.0]]}
            }  # although called combo_field2, it is not a multimodal_tensor field in the existing doc
        }, {})

        self.assertEquals([], combo_field2.chunks)
        self.assertEquals([], combo_field2.embeddings)

    def test_populate_tensor_from_existing_docs_will_not_populate_multimodal_field_with_different_weight(self):
        combo_field2 = self._get_combo_field2()

        self.container.populate_tensor_from_existing_doc({
            MARQO_DOC_ID: 'doc_id1',
            'tensor_field2': 'tensor_field2_content',
            'subfield1': 'subfield1_content',
            MARQO_DOC_TENSORS: {
                'combo_field2': {MARQO_DOC_CHUNKS: ['combo_field2_content'], MARQO_DOC_EMBEDDINGS: [[1.0, 2.0]]}
            }
        }, {'combo_field2': {'subfield1': 0.5, 'tensor_field2': 5.0}})  # weight is different

        self.assertEquals([], combo_field2.chunks)
        self.assertEquals([], combo_field2.embeddings)

    def test_populate_tensor_from_existing_docs_will_not_populate_multimodal_field_with_different_subfields(self):
        combo_field2 = self._get_combo_field2()

        self.container.populate_tensor_from_existing_doc({
            MARQO_DOC_ID: 'doc_id1',
            'tensor_field2': 'changed_tensor_field2_content',  # content of this field is changed
            'subfield1': 'subfield1_content',
            MARQO_DOC_TENSORS: {
                'combo_field2': {MARQO_DOC_CHUNKS: ['combo_field2_content'], MARQO_DOC_EMBEDDINGS: [[1.0, 2.0]]}
            }
        }, {'combo_field2': {'subfield1': 2.0, 'tensor_field2': 5.0}})

        self.assertEquals([], combo_field2.chunks)
        self.assertEquals([], combo_field2.embeddings)

    def test_populate_tensor_from_existing_docs_should_populate_multimodal_field_if_all_conditions_match(self):
        combo_field2 = self._get_combo_field2()

        self.container.populate_tensor_from_existing_doc({
            MARQO_DOC_ID: 'doc_id1',
            'tensor_field2': 'tensor_field2_content',
            'subfield1': 'subfield1_content',
            MARQO_DOC_TENSORS: {
                'combo_field2': {MARQO_DOC_CHUNKS: ['combo_field2_content'], MARQO_DOC_EMBEDDINGS: [[1.0, 2.0]]}
            }  # although called combo_field2, it is not a multimodal_tensor field in the existing doc
        }, {'combo_field2': {'subfield1': 2.0, 'tensor_field2': 5.0}})

        self.assertEquals(['combo_field2_content'], combo_field2.chunks)
        self.assertEquals([[1.0, 2.0]], combo_field2.embeddings)

    def _get_combo_field2(self):
        self.container.collect('doc_id1', 'tensor_field2', 'tensor_field2_content', FieldType.Text)
        self.container.collect('doc_id1', 'subfield1', 'subfield1_content', FieldType.Text)
        list(self.container.collect_multi_modal_fields('doc_id1', True))
        return self.container._tensor_field_map['doc_id1']['combo_field2']

    def test_traversing_tensor_fields_to_vectorise_should_return_all_fields(self):
        self.container.collect('doc_id1', 'tensor_field1', 'tensor_field1_content', FieldType.Text)
        self.container.collect('doc_id1', 'tensor_field2', 'tensor_field2_content', FieldType.ImagePointer)
        self.container.collect('doc_id1', 'subfield1', 'subfield1_content', FieldType.Text)
        list(self.container.collect_multi_modal_fields('doc_id1', True))
        self.container.collect('doc_id2', 'tensor_field1', 'tensor_field1_content', FieldType.AudioPointer)
        self.container.collect('doc_id2', 'tensor_field2', 'tensor_field2_content', FieldType.VideoPointer)
        list(self.container.collect_multi_modal_fields('doc_id2', True))

        fields = list(self.container.tensor_fields_to_vectorise(FieldType.Text, FieldType.ImagePointer,
                                                                FieldType.VideoPointer, FieldType.AudioPointer))
        self.assertIn(('doc_id1', 'tensor_field1', self.container._tensor_field_map['doc_id1']['tensor_field1']), fields)
        self.assertIn(('doc_id1', 'tensor_field2', self.container._tensor_field_map['doc_id1']['tensor_field2']), fields)
        self.assertIn(('doc_id1', 'subfield1', self.container._tensor_field_map['doc_id1']['subfield1']), fields)
        self.assertIn(('doc_id2', 'tensor_field1', self.container._tensor_field_map['doc_id2']['tensor_field1']), fields)
        self.assertIn(('doc_id2', 'tensor_field2', self.container._tensor_field_map['doc_id2']['tensor_field2']), fields)

    def test_traversing_tensor_fields_to_vectorise_skips_resolved_fields(self):
        self.container.collect('doc_id1', 'tensor_field1', 'tensor_field1_content', FieldType.Text)
        self.container.collect('doc_id1', 'subfield1', 'subfield1_content', FieldType.Text)
        list(self.container.collect_multi_modal_fields('doc_id1', True))

        # resolve tensor_field1
        tensor_field1 = self.container._tensor_field_map['doc_id1']['tensor_field1']
        tensor_field1.populate_chunks_and_embeddings(['hello world'], [[1.0, 1.2]])

        fields = list(self.container.tensor_fields_to_vectorise(FieldType.Text))
        self.assertEquals(1, len(fields))
        (doc_id, field_name, _) = fields[0]
        self.assertEquals('doc_id1', doc_id)
        self.assertEquals('subfield1', field_name)

    def test_traversing_tensor_fields_to_vectorise_skips_removed_doc(self):
        self.container.collect('doc_id1', 'tensor_field1', 'tensor_field1_content', FieldType.Text)
        self.container.collect('doc_id1', 'tensor_field2', 'tensor_field2_content', FieldType.Text)
        self.container.collect('doc_id2', 'tensor_field1', 'tensor_field1_content', FieldType.Text)
        self.container.collect('doc_id2', 'tensor_field2', 'tensor_field2_content', FieldType.Text)

        fields = []
        for doc_id, field_name, _ in self.container.tensor_fields_to_vectorise(FieldType.Text):
            fields.append((doc_id, field_name))
            if doc_id == 'doc_id1':  # after taking in the first field of doc_id1, remove doc_id1 to simulate a failure
                self.container.remove_doc(doc_id)

        self.assertEquals(3, len(fields))
        self.assertIn(('doc_id1', 'tensor_field1'), fields)
        self.assertIn(('doc_id2', 'tensor_field1'), fields)
        self.assertIn(('doc_id2', 'tensor_field2'), fields)

    def test_traversing_tensor_fields_to_vectorise_by_type(self):
        self.container.collect('doc_id1', 'tensor_field1', 'tensor_field1_content', FieldType.Text)
        self.container.collect('doc_id1', 'tensor_field2', 'tensor_field2_content', FieldType.ImagePointer)

        fields = list(self.container.tensor_fields_to_vectorise(FieldType.ImagePointer))
        self.assertEquals(1, len(fields))
        (doc_id, field_name, _) = fields[0]
        self.assertEquals('doc_id1', doc_id)
        self.assertEquals('tensor_field2', field_name)

    def test_traversing_tensor_fields_to_vectorise_skips_subfields_for_resolved_multimodal_fields(self):
        self.container.collect('doc_id1', 'tensor_field1', 'tensor_field1_content', FieldType.Text)
        self.container.collect('doc_id1', 'tensor_field2', 'tensor_field2_content', FieldType.Text)
        self.container.collect('doc_id1', 'subfield1', 'subfield1_content', FieldType.Text)
        list(self.container.collect_multi_modal_fields('doc_id1', True))

        tensor_field2 = self.container._tensor_field_map['doc_id1']['tensor_field2']
        tensor_field2.populate_chunks_and_embeddings(['hello world'], [[1.0, 1.2]])
        combo_field1 = self.container._tensor_field_map['doc_id1']['combo_field1']
        combo_field1.populate_chunks_and_embeddings(['hello world'], [[1.0, 1.2]])
        combo_field2 = self.container._tensor_field_map['doc_id1']['combo_field2']
        combo_field2.populate_chunks_and_embeddings(['hello world'], [[1.0, 1.2]])

        fields = list(self.container.tensor_fields_to_vectorise(FieldType.Text))
        self.assertEquals(1, len(fields))
        self.assertEquals('tensor_field1', fields[0][1])

        # subfield 1 does not need to be vectorised since all the combo fields using it are resolved
        # tensor_fields2 does not need to be vectorised since its embeddings are populated and the combo field that
        #   needs it is already resolved

    def test_get_tensor_field_content_for_persisting(self):
        self.container.collect('doc_id1', 'tensor_field1', 'tensor_field1_content', FieldType.Text)
        self.container.collect('doc_id1', 'tensor_field2', 'tensor_field2_content', FieldType.Text)
        self.container.collect('doc_id1', 'subfield1', 'subfield1_content', FieldType.Text)  # subfield is not persisted
        list(self.container.collect_multi_modal_fields('doc_id1', True))

        self.container._tensor_field_map['doc_id1']['tensor_field1'].populate_chunks_and_embeddings(['hello world'], [[1.0, 1.2]])
        self.container._tensor_field_map['doc_id1']['tensor_field2'].populate_chunks_and_embeddings(['hello world'], [[1.0, 1.2]])

        fields = self.container.get_tensor_field_content('doc_id1')
        self.assertEquals(4, len(fields))
        self.assertIn('tensor_field1', fields)
        self.assertIn('tensor_field2', fields)
        self.assertIn('combo_field1', fields)
        self.assertIn('combo_field2', fields)

    def test_get_tensor_field_content_for_persisting_skips_multimodal_field_with_no_subfields(self):
        self.container.collect('doc_id1', 'tensor_field1', 'tensor_field1_content', FieldType.Text)
        self.container.collect('doc_id1', 'tensor_field2', 'tensor_field2_content', FieldType.Text)
        list(self.container.collect_multi_modal_fields('doc_id1', True))

        self.container._tensor_field_map['doc_id1']['tensor_field1'].populate_chunks_and_embeddings(['hello world'], [[1.0, 1.2]])
        self.container._tensor_field_map['doc_id1']['tensor_field2'].populate_chunks_and_embeddings(['hello world'], [[1.0, 1.2]])

        fields = self.container.get_tensor_field_content('doc_id1')
        self.assertEquals(3, len(fields))
        self.assertIn('tensor_field1', fields)
        self.assertIn('tensor_field2', fields)
        self.assertIn('combo_field2', fields)  # combo_field2 has tensor_field2 as subfield
        # combo_field1 has subfield1 as the only subfield, since subfield1 is not present, combo_field1 does not
        # have content either
