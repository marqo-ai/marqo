import unittest
from typing import cast

from marqo.core.constants import MARQO_DOC_ID, MARQO_DOC_TENSORS, MARQO_DOC_CHUNKS, MARQO_DOC_EMBEDDINGS
from marqo.core.exceptions import AddDocumentsError
from marqo.core.models.marqo_index import FieldType
from marqo.core.inference.tensor_fields_container import TensorFieldsContainer, MultiModalTensorField


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

        self.assertEqual({'subfield1', 'tensor_field2'}, self.container.get_multimodal_sub_fields())
        self.assertEqual({'subfield1': 1.0}, self.container.get_multimodal_field_mapping('combo_field1'))
        self.assertEqual({'subfield1': 2.0, 'tensor_field2': 5.0},
                          self.container.get_multimodal_field_mapping('combo_field2'))

        self.assertEqual(0, len(self.container._tensor_field_map))

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
                content = self.container.collect(
                    'doc_id1', 'field1', field_content, field_type
                )
                self.assertEqual(field_content, content)
                # verify that they won't be collected to tensor field maps
                self.assertEqual(0, len(self.container._tensor_field_map))

    def test_collect_custom_vector_field(self):
        content = self.container.collect('doc_id1', 'custom_vector_field1', {
            'content': 'content1',
            'vector': [1.0, 2.0]
        })

        self.assertEqual('content1', content)
        self.assertIn('doc_id1', self.container._tensor_field_map)
        self.assertIn('custom_vector_field1', self.container._tensor_field_map['doc_id1'])

        field = self.container._tensor_field_map['doc_id1']['custom_vector_field1']
        self.assertEqual('content1', field.field_content)
        self.assertEqual(FieldType.CustomVector, field.field_type)
        self.assertEqual(['content1'], field.chunks)
        self.assertEqual([[0.4472135954999579, 0.8944271909999159]], field.embeddings)  # normalised
        self.assertTrue(field.is_top_level_tensor_field)
        self.assertFalse(field.is_multimodal_subfield)

    def test_collect_custom_vector_field_should_fail_with_zero_magnitude_vector(self):
        with self.assertRaises(AddDocumentsError) as context:
            self.container.collect('doc_id1', 'custom_vector_field1', {
                'content': 'content1',
                'vector': [0.0, 0.0]
            })
        self.assertIn('Field custom_vector_field1 has zero magnitude vector, cannot normalize.',
                      str(context.exception))

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
        for field_type in [
            FieldType.Text,
            FieldType.ImagePointer,
            FieldType.AudioPointer,
            FieldType.VideoPointer,
            None  # unstructured does not have field type passed in
        ]:
            with self.subTest(msg=f'field_type {field_type}'):
                content = self.container.collect('doc_id1', 'tensor_field1', 'content', field_type)
                self.assertEqual('content', content)
                self.assertIn('doc_id1', self.container._tensor_field_map)
                self.assertIn('tensor_field1', self.container._tensor_field_map['doc_id1'])

                field = self.container._tensor_field_map['doc_id1']['tensor_field1']
                self.assertEqual('content', field.field_content)
                self.assertEqual(field_type, field.field_type)
                self.assertIsNone(field.chunks)
                self.assertIsNone(field.embeddings)
                self.assertIsNone(field.multimodal_subfield_embedding)
                self.assertTrue(field.is_top_level_tensor_field)
                self.assertFalse(field.is_multimodal_subfield)

    def test_collect_tensor_field_can_identify_toplevel_or_subfield(self):
        test_cases = [
            # field_name, is_tensor_field, is_multimodal_subfield
            ('tensor_field1', True, False),
            ('tensor_field2', True, True),
            ('subfield1', False, True),
        ]

        for (field_name, is_tensor_field, is_multimodal_subfield) in test_cases:
            with self.subTest(msg=f'{field_name}: is_tensor_field={is_tensor_field}, '
                                  f'is_multimodal_subfield={is_multimodal_subfield}'):

                self.container.collect('doc_id1', field_name, 'content')

                tensor_field_content = self.container._tensor_field_map['doc_id1'][field_name]
                self.assertEqual(is_tensor_field, tensor_field_content.is_top_level_tensor_field)
                self.assertEqual(is_multimodal_subfield, tensor_field_content.is_multimodal_subfield)

    def test_remove_doc(self):
        self.container.collect('doc_id1', 'tensor_field1', 'content')
        self.container.collect('doc_id1', 'tensor_field2', 'content')
        self.container.collect('doc_id2', 'tensor_field2', 'content')

        self.assertIn('doc_id1', self.container._tensor_field_map)
        self.assertIn('doc_id2', self.container._tensor_field_map)
        self.assertNotIn('doc_id3', self.container._tensor_field_map)

        self.container.remove_doc('doc_id1')
        self.container.remove_doc('doc_id3')

        self.assertNotIn('doc_id1', self.container._tensor_field_map)
        self.assertIn('doc_id2', self.container._tensor_field_map)

    def test_collect_multimodal_fields_should_return_all(self):
        fields = list(self.container.collect_multi_modal_fields('doc_id1', True))
        self.assertEqual(('combo_field1', {'subfield1': 1.0}), fields[0])
        self.assertEqual(('combo_field2', {'subfield1': 2.0, 'tensor_field2': 5.0}), fields[1])

    def test_collect_multimodal_fields_should_populate_subfields(self):
        self.container.collect('doc_id1', 'tensor_field2', 'tensor_field2_content')
        self.container.collect('doc_id1', 'subfield1', 'subfield1_content')

        list(self.container.collect_multi_modal_fields('doc_id1', True))

        self.assertIn('doc_id1', self.container._tensor_field_map)
        self.assertIn('combo_field1', self.container._tensor_field_map['doc_id1'])

        combo_field1 = cast(MultiModalTensorField, self.container._tensor_field_map['doc_id1']['combo_field1'])
        self.assertEqual(FieldType.MultimodalCombination, combo_field1.field_type)
        self.assertEqual('', combo_field1.field_content)
        self.assertTrue(combo_field1.is_top_level_tensor_field)
        self.assertFalse(combo_field1.is_multimodal_subfield)
        self.assertEqual({'subfield1': 1.0}, combo_field1.weights)
        self.assertEqual({'subfield1': self.container._tensor_field_map['doc_id1']['subfield1']},
                         combo_field1.subfields)
        self.assertTrue(combo_field1.normalize_embeddings)

        combo_field2 = cast(MultiModalTensorField, self.container._tensor_field_map['doc_id1']['combo_field2'])
        self.assertEqual({'subfield1': self.container._tensor_field_map['doc_id1']['subfield1'],
                         'tensor_field2': self.container._tensor_field_map['doc_id1']['tensor_field2']},
                         combo_field2.subfields)

    def test_collect_multimodal_fields_should_not_populate_non_existing_subfields(self):
        self.container.collect('doc_id1', 'tensor_field2', 'tensor_field2_content')

        list(self.container.collect_multi_modal_fields('doc_id1', True))

        combo_field1 = cast(MultiModalTensorField, self.container._tensor_field_map['doc_id1']['combo_field1'])
        self.assertEqual({}, combo_field1.subfields)

        combo_field2 = cast(MultiModalTensorField, self.container._tensor_field_map['doc_id1']['combo_field2'])
        self.assertEqual({'tensor_field2': self.container._tensor_field_map['doc_id1']['tensor_field2']},
                         combo_field2.subfields)

    def test_populate_tensor_from_existing_docs_will_not_populate_if_doc_id_does_not_match(self):
        self.container.collect('doc_id1', 'tensor_field1', 'tensor_field1_content')
        tensor_field1 = self.container._tensor_field_map['doc_id1']['tensor_field1']

        self.container.populate_tensor_from_existing_doc({
            MARQO_DOC_ID: 'doc_id2',
            'tensor_field1': 'tensor_field1_content',
            MARQO_DOC_TENSORS: {
                'tensor_field1': {MARQO_DOC_CHUNKS: ['tensor_field1_content'], MARQO_DOC_EMBEDDINGS: [[1.0, 2.0]]}
            }
        }, {})

        self.assertIsNone(tensor_field1.chunks)
        self.assertIsNone(tensor_field1.embeddings)

    def test_populate_tensor_from_existing_docs_should_populate_if_doc_id_matches(self):
        self.container.collect('doc_id1', 'tensor_field1', 'tensor_field1_content')
        tensor_field1 = self.container._tensor_field_map['doc_id1']['tensor_field1']

        self.container.populate_tensor_from_existing_doc({
            MARQO_DOC_ID: 'doc_id1',
            'tensor_field1': 'tensor_field1_content',
            MARQO_DOC_TENSORS: {
                'tensor_field1': {MARQO_DOC_CHUNKS: ['tensor_field1_content'], MARQO_DOC_EMBEDDINGS: [[1.0, 2.0]]}
            }
        }, {})

        self.assertEqual(['tensor_field1_content'], tensor_field1.chunks)
        self.assertEqual([[1.0, 2.0]], tensor_field1.embeddings)

    def test_populate_tensor_from_existing_docs_will_not_populate_if_content_changes(self):
        self.container.collect('doc_id1', 'tensor_field1', 'changed_content')
        tensor_field1 = self.container._tensor_field_map['doc_id1']['tensor_field1']

        self.container.populate_tensor_from_existing_doc({
            MARQO_DOC_ID: 'doc_id1',
            'tensor_field1': 'tensor_field1_content',
            MARQO_DOC_TENSORS: {
                'tensor_field1': {MARQO_DOC_CHUNKS: ['tensor_field1_content'], MARQO_DOC_EMBEDDINGS: [[1.0, 2.0]]}
            }
        }, {})

        self.assertIsNone(tensor_field1.chunks)
        self.assertIsNone(tensor_field1.embeddings)

    def test_populate_tensor_from_existing_docs_will_not_populate_if_field_does_not_exist(self):
        self.container.collect('doc_id1', 'tensor_field1', 'tensor_field1_content')
        tensor_field1 = self.container._tensor_field_map['doc_id1']['tensor_field1']

        self.container.populate_tensor_from_existing_doc({
            MARQO_DOC_ID: 'doc_id1',
            'tensor_field2': 'tensor_field2_content',  # tensor_field1 does not exist in the existing doc
            MARQO_DOC_TENSORS: {}
        }, {})

        self.assertIsNone(tensor_field1.chunks)
        self.assertIsNone(tensor_field1.embeddings)

    def test_populate_tensor_from_existing_docs_will_not_populate_if_embedding_does_not_exist(self):
        self.container.collect('doc_id1', 'tensor_field1', 'tensor_field1_content')
        tensor_field1 = self.container._tensor_field_map['doc_id1']['tensor_field1']

        self.container.populate_tensor_from_existing_doc({
            MARQO_DOC_ID: 'doc_id1',
            'tensor_field1': 'tensor_field1_content',
            MARQO_DOC_TENSORS: {}  # embedding for tensor_field1 does not exist in the existing doc
        }, {})

        self.assertIsNone(tensor_field1.chunks)
        self.assertIsNone(tensor_field1.embeddings)

    def test_populate_tensor_from_existing_docs_will_not_populate_if_existing_field_is_multimodal_combo_field(self):
        self.container.collect('doc_id1', 'tensor_field1', 'tensor_field1_content')
        tensor_field1 = self.container._tensor_field_map['doc_id1']['tensor_field1']

        self.container.populate_tensor_from_existing_doc({
            MARQO_DOC_ID: 'doc_id1',
            MARQO_DOC_TENSORS: {
                'tensor_field1': {MARQO_DOC_CHUNKS: ['tensor_field1_content'], MARQO_DOC_EMBEDDINGS: [[1.0, 2.0]]}
            }
        }, {'tensor_field1': {'subfield1': 1.0}})  # tensor_field1 is a multimodal combo field

        self.assertIsNone(tensor_field1.chunks)
        self.assertIsNone(tensor_field1.embeddings)

    def test_populate_tensor_from_existing_docs_will_not_populate_for_custom_vector_field(self):
        self.container.collect('doc_id1', 'custom_vector_field1', {
            'content': 'content1',
            'vector': [1.0, 2.0]
        })
        custom_vector_field1 = self.container._tensor_field_map['doc_id1']['custom_vector_field1']

        self.container.populate_tensor_from_existing_doc({
            MARQO_DOC_ID: 'doc_id1',
            'custom_vector_field1': 'content1',
            MARQO_DOC_TENSORS: {
                'custom_vector_field1': {MARQO_DOC_CHUNKS: ['content2'], MARQO_DOC_EMBEDDINGS: [[3.0, 4.0]]}
            }  # embedding for tensor_field1 does not exist in the existing doc
        }, {})

        self.assertEqual(['content1'], custom_vector_field1.chunks)
        self.assertEqual([[0.4472135954999579, 0.8944271909999159]], custom_vector_field1.embeddings)

    def test_populate_tensor_from_existing_docs_will_not_populate_for_multimodal_field_if_it_does_not_exist(self):
        combo_field2 = self._get_combo_field2()

        self.container.populate_tensor_from_existing_doc({
            MARQO_DOC_ID: 'doc_id1',
            'tensor_field2': 'tensor_field2_content',
            'subfield1': 'subfield1_content',
            MARQO_DOC_TENSORS: {}  # embedding for combo_field2 does not exist in the existing doc
        }, {})

        self.assertIsNone(combo_field2.chunks)
        self.assertIsNone(combo_field2.embeddings)

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

        self.assertIsNone(combo_field2.chunks)
        self.assertIsNone(combo_field2.embeddings)

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

        self.assertIsNone(combo_field2.chunks)
        self.assertIsNone(combo_field2.embeddings)

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

        self.assertIsNone(combo_field2.chunks)
        self.assertIsNone(combo_field2.embeddings)

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

        self.assertEqual(['combo_field2_content'], combo_field2.chunks)
        self.assertEqual([[1.0, 2.0]], combo_field2.embeddings)

    def _get_combo_field2(self):
        self.container.collect('doc_id1', 'tensor_field2', 'tensor_field2_content')
        self.container.collect('doc_id1', 'subfield1', 'subfield1_content')
        list(self.container.collect_multi_modal_fields('doc_id1', True))
        return self.container._tensor_field_map['doc_id1']['combo_field2']

    def test_select_unresolved_tensor_fields_should_return_all_fields(self):
        self.container.collect('doc_id1', 'tensor_field1', 'tensor_field1_content')
        self.container.collect('doc_id1', 'tensor_field2', 'tensor_field2_content')
        self.container.collect('doc_id1', 'subfield1', 'subfield1_content')
        list(self.container.collect_multi_modal_fields('doc_id1', True))
        self.container.collect('doc_id2', 'tensor_field1', 'tensor_field1_content')
        self.container.collect('doc_id2', 'tensor_field2', 'tensor_field2_content')
        list(self.container.collect_multi_modal_fields('doc_id2', True))

        fields = self.container.select_unresolved_tensor_fields()
        self.assertListEqual(fields, [
            self.container._tensor_field_map['doc_id1']['tensor_field1'],
            self.container._tensor_field_map['doc_id1']['tensor_field2'],
            self.container._tensor_field_map['doc_id1']['subfield1'],
            self.container._tensor_field_map['doc_id2']['tensor_field1'],
            self.container._tensor_field_map['doc_id2']['tensor_field2'],
        ])

    def test_select_unresolved_tensor_fields_skips_resolved_fields(self):
        self.container.collect('doc_id1', 'tensor_field1', 'tensor_field1_content')
        self.container.collect('doc_id1', 'subfield1', 'subfield1_content')
        list(self.container.collect_multi_modal_fields('doc_id1', True))

        # resolve tensor_field1
        tensor_field1 = self.container._tensor_field_map['doc_id1']['tensor_field1']
        tensor_field1.populate_chunks_and_embeddings(['hello world'], [[1.0, 1.2]])

        fields = self.container.select_unresolved_tensor_fields()
        self.assertListEqual(fields, [self.container._tensor_field_map['doc_id1']['subfield1']])

    def test_select_unresolved_tensor_fields_skips_custom_fields_and_multi_modal_fields(self):
        self.container.collect('doc_id1', 'tensor_field1', 'tensor_field1_content')
        self.container.collect('doc_id1', 'tensor_field2', 'tensor_field2_content')
        self.container.collect('doc_id1', 'custom_vector_field1', {
            'content': 'content1',
            'vector': [1.0, 2.0]
        })
        list(self.container.collect_multi_modal_fields('doc_id1', True))

        fields = self.container.select_unresolved_tensor_fields()
        self.assertListEqual(fields, [
            self.container._tensor_field_map['doc_id1']['tensor_field1'],
            self.container._tensor_field_map['doc_id1']['tensor_field2'],
        ])

    def test_select_unresolved_tensor_fields_skips_fields_not_matching_predicates(self):
        self.container.collect('doc_id1', 'tensor_field1', 'tensor_field1_content')
        self.container.collect('doc_id1', 'tensor_field2', 'tensor_field2_content')
        self.container.collect('doc_id2', 'tensor_field1', 'tensor_field2_content')

        fields = self.container.select_unresolved_tensor_fields(lambda f: f.field_name == 'tensor_field1')
        self.assertListEqual(fields, [
            self.container._tensor_field_map['doc_id1']['tensor_field1'],
            self.container._tensor_field_map['doc_id2']['tensor_field1'],
        ])

    def test_get_tensor_field_content_for_persisting(self):
        self.container.collect('doc_id1', 'tensor_field1', 'tensor_field1_content')
        self.container.collect('doc_id1', 'tensor_field2', 'tensor_field2_content')
        self.container.collect('doc_id1', 'subfield1', 'subfield1_content')  # subfield is not persisted
        list(self.container.collect_multi_modal_fields('doc_id1', True))

        self.container._tensor_field_map['doc_id1']['tensor_field1'].populate_chunks_and_embeddings(['hello world'], [[1.0, 1.2]])
        self.container._tensor_field_map['doc_id1']['tensor_field2'].populate_chunks_and_embeddings(['hello world'], [[1.0, 1.2]])

        fields = self.container.get_tensor_field_content('doc_id1')
        self.assertEqual(4, len(fields))
        self.assertIn('tensor_field1', fields)
        self.assertIn('tensor_field2', fields)
        self.assertIn('combo_field1', fields)
        self.assertIn('combo_field2', fields)

    def test_get_tensor_field_content_for_persisting_skips_multimodal_field_with_no_subfields(self):
        self.container.collect('doc_id1', 'tensor_field1', 'tensor_field1_content')
        self.container.collect('doc_id1', 'tensor_field2', 'tensor_field2_content')
        list(self.container.collect_multi_modal_fields('doc_id1', True))

        self.container._tensor_field_map['doc_id1']['tensor_field1'].populate_chunks_and_embeddings(['hello world'], [[1.0, 1.2]])
        self.container._tensor_field_map['doc_id1']['tensor_field2'].populate_chunks_and_embeddings(['hello world'], [[1.0, 1.2]])

        fields = self.container.get_tensor_field_content('doc_id1')
        self.assertEqual(3, len(fields))
        self.assertIn('tensor_field1', fields)
        self.assertIn('tensor_field2', fields)
        self.assertIn('combo_field2', fields)  # combo_field2 has tensor_field2 as subfield
        # combo_field1 has subfield1 as the only subfield, since subfield1 is not present, combo_field1 does not
        # have content either

    def test_has_unresolved_parent_field_returns_false_when_field_is_not_a_subfield(self):
        self.container.collect('doc_id1', 'tensor_field1', 'tensor_field1_content')

        field = self.container._tensor_field_map['doc_id1']['tensor_field1']
        self.assertFalse(self.container.has_unresolved_parent_field(field))

    def test_has_unresolved_parent_field_returns_false_when_field_has_unresolved_parents_fields(self):
        self.container.collect('doc_id1', 'subfield1', 'tensor_field1_content')
        self.container.collect('doc_id1', 'tensor_field2', 'tensor_field2_content')
        # combo_field1 and combo_field2 both uses tensor_field1
        list(self.container.collect_multi_modal_fields('doc_id1', True))

        # we resolve one of the combo field
        self.container._tensor_field_map['doc_id1']['combo_field2'].populate_chunks_and_embeddings(
            ['hello world'], [[1.0, 1.2]])

        field1 = self.container._tensor_field_map['doc_id1']['subfield1']
        self.assertTrue(self.container.has_unresolved_parent_field(field1))

        # instead, tensor_field2 is only used by combo_field2, so it should not have any unresolved parent
        field2 = self.container._tensor_field_map['doc_id1']['tensor_field2']
        self.assertFalse(self.container.has_unresolved_parent_field(field2))

    def test_has_unresolved_parent_field_returns_true_when_all_parents_fields_are_resolved(self):
        self.container.collect('doc_id1', 'subfield1', 'tensor_field1_content')
        self.container.collect('doc_id1', 'tensor_field2', 'tensor_field2_content')
        # combo_field1 and combo_field2 both uses tensor_field1
        list(self.container.collect_multi_modal_fields('doc_id1', True))

        # we resolve both combo fields
        self.container._tensor_field_map['doc_id1']['combo_field1'].populate_chunks_and_embeddings(
            ['hello world'], [[1.0, 1.2]])
        self.container._tensor_field_map['doc_id1']['combo_field2'].populate_chunks_and_embeddings(
            ['hello world'], [[1.0, 1.2]])

        field1 = self.container._tensor_field_map['doc_id1']['subfield1']
        self.assertFalse(self.container.has_unresolved_parent_field(field1))

        field2 = self.container._tensor_field_map['doc_id1']['tensor_field2']
        self.assertFalse(self.container.has_unresolved_parent_field(field2))
