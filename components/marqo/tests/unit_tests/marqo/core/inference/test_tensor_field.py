import unittest

import numpy as np

from marqo.core.inference.api import Modality
from marqo.core.models.marqo_index import FieldType
from marqo.core.inference.tensor_fields_container import TensorField, MultiModalTensorField


class TestTensorField(unittest.TestCase):
    def test_is_unresolved_top_level_field(self):
        """
        Test if TensorField.is_unresolved_top_level_field returns the expected result
        """
        test_cases = [
            # is_top_level_field, embeddings, expected_result
            (True, None, True),  # embedding is None
            (True, [], True),    # embedding is empty list
            (True, [[1.0, 2.0]], False),  # embedding is present

            # if not top level tensor field, should always return false
            (False, None, False),
            (False, [], False),
            (False, [[1.0, 2.0]], False),
        ]

        for is_top_level_field, embeddings, expected_result in test_cases:
            with self.subTest(f'top_level: {is_top_level_field}, embeddings: {embeddings}'):
                field = TensorField(
                    doc_id='id', field_name='field', field_content='content', modality=Modality.TEXT,
                    is_top_level_tensor_field=is_top_level_field, is_multimodal_subfield=True,
                    embeddings=embeddings
                )

                self.assertEqual(expected_result, field.is_unresolved_top_level_field())

    def test_is_unresolved_multimodal_subfield(self):
        """
        Test if TensorField.is_unresolved_multimodal_subfield returns the expected result
        """

        test_cases = [
            # modality, is_subfield, embeddings, subfield_embedding, expected_result
            (Modality.TEXT, True, None, None, True),
            (Modality.TEXT, True, [[1.0, 2.0]], None, True),  # text subfield, with chunked embeddings populated
            (Modality.TEXT, True, None, [1.0, 2.0], False),  # text subfield, with subfield embeddings populated
            (Modality.TEXT, False, None, None, False),   # not subfield

            (Modality.IMAGE, True, None, None, True),
            (Modality.IMAGE, True, [[1.0, 2.0]], None, True),  # image subfield, with chunked embeddings populated
            (Modality.IMAGE, True, None, [1.0, 2.0], False),  # image subfield, with subfield embeddings populated
            (Modality.IMAGE, False, None, None, False),  # not subfield

            (Modality.AUDIO, True, None, None, True),
            (Modality.AUDIO, True, [[1.0, 2.0]], None, False),  # audio subfield, with chunked embeddings populated
            (Modality.AUDIO, False, None, None, False),  # not subfield

            (Modality.VIDEO, True, None, None, True),
            (Modality.VIDEO, True, [[1.0, 2.0]], None, False),  # video subfield, with chunked embeddings populated
            (Modality.VIDEO, False, None, None, False),  # not subfield
        ]

        for modality, is_subfield, embeddings, subfield_embedding, expected_result in test_cases:
            with self.subTest(f'{modality.value}, subfield: {is_subfield}, embeddings: {embeddings}, '
                              f'subfield_embedding: {subfield_embedding}'):
                field = TensorField(
                    doc_id='id', field_name='field', field_content='content', modality=modality,
                    is_top_level_tensor_field=True, is_multimodal_subfield=is_subfield,
                    embeddings=embeddings, multimodal_subfield_embedding=subfield_embedding
                )
                self.assertEqual(expected_result, field.is_unresolved_multimodal_subfield())

    def test_tensor_field_chunks_and_embeddings(self):
        chunks = ['abc']
        embeddings = [[1.0, 2.0]]
        field = TensorField(
            doc_id='id', field_name='field', field_content='content',
            is_top_level_tensor_field=True, chunks=chunks, embeddings=embeddings
        )

        self.assertEqual(chunks, field.tensor_field_chunks)
        self.assertEqual(embeddings, field.tensor_field_embeddings)

    def test_tensor_field_chunk_and_embedding_should_raise_error_if_field_is_not_top_level(self):
        field = TensorField(
            doc_id='id', field_name='field', field_content='content',
            is_top_level_tensor_field=False, is_multimodal_subfield=True
        )

        with self.assertRaises(ValueError) as context:
            self.assertIsNone(field.tensor_field_chunks)
        self.assertIn('field of doc: id is not a top level tensor field', str(context.exception))

        with self.assertRaises(ValueError) as context:
            self.assertIsNone(field.tensor_field_embeddings)
        self.assertIn('field of doc: id is not a top level tensor field', str(context.exception))

    def test_subfield_chunk_and_embedding_should_raise_error_if_field_is_not_subfield(self):
        field = TensorField(
            doc_id='id', field_name='field', field_content='content',
            is_top_level_tensor_field=True, is_multimodal_subfield=False
        )

        with self.assertRaises(ValueError) as context:
            self.assertIsNone(field.subfield_chunk)
        self.assertIn('field of doc: id is not a subfield', str(context.exception))

        with self.assertRaises(ValueError) as context:
            self.assertIsNone(field.subfield_embedding)
        self.assertIn('field of doc: id is not a subfield', str(context.exception))

    def test_subfield_chunk_and_embedding_for_text_field(self):
        multimodal_subfield_embedding = [1.0, 2.0]
        field = TensorField(
            doc_id='id', field_name='field', field_content='content', modality=Modality.TEXT,
            is_top_level_tensor_field=False, is_multimodal_subfield=True,
            multimodal_subfield_embedding=multimodal_subfield_embedding
        )

        self.assertEqual('content', field.subfield_chunk)
        self.assertEqual(multimodal_subfield_embedding, field.subfield_embedding)

    def test_subfield_chunk_and_embedding_for_image_field(self):
        multimodal_subfield_embedding = [1.0, 2.0]
        field = TensorField(
            doc_id='id', field_name='field', field_content='http://a.com/c.jpg', modality=Modality.IMAGE,
            is_top_level_tensor_field=False, is_multimodal_subfield=True,
            multimodal_subfield_embedding=multimodal_subfield_embedding
        )

        self.assertEqual('http://a.com/c.jpg', field.subfield_chunk)
        self.assertEqual(multimodal_subfield_embedding, field.subfield_embedding)

    def test_subfield_chunk_and_embedding_for_audio_field(self):
        embeddings = [[1.0, 2.0], [3.0, 4.0]]
        field = TensorField(
            doc_id='id', field_name='field', field_content='http://a.com/c.mp3', modality=Modality.AUDIO,
            is_top_level_tensor_field=False, is_multimodal_subfield=True,
            embeddings=embeddings
        )

        self.assertEqual('http://a.com/c.mp3', field.subfield_chunk)
        # the mean value of all arrays in the embedding
        self.assertEqual([2.0, 3.0], field.subfield_embedding)

    def test_subfield_chunk_and_embedding_for_video_field(self):
        embeddings = [[1.0, 2.0], [3.0, 4.0]]
        field = TensorField(
            doc_id='id', field_name='field', field_content='http://a.com/c.mp4', modality=Modality.VIDEO,
            is_top_level_tensor_field=False, is_multimodal_subfield=True,
            embeddings=embeddings
        )

        self.assertEqual('http://a.com/c.mp4', field.subfield_chunk)
        # the mean value of all arrays in the embedding
        self.assertEqual([2.0, 3.0], field.subfield_embedding)


class TestMultiModalTensorField(unittest.TestCase):
    def test_multimodal_combo_field_with_no_subfield_returns_empty_chunks_and_embeddings(self):
        multimodal_field = MultiModalTensorField(
            doc_id='id',
            field_name='combo1',
            weights={'field_1': 1.0, 'field_2': 2.0},
            field_content='',
            field_type=FieldType.MultimodalCombination,
            subfields={},
            is_top_level_tensor_field=True,
            normalize_embeddings=True
        )

        self.assertEqual([], multimodal_field.tensor_field_chunks)
        self.assertEqual([], multimodal_field.tensor_field_embeddings)

    def test_multimodal_combo_field_derived_from_one_subfield(self):
        vector_chunk = np.array([1.0, 2.0])
        test_cases = [
            (False, [vector_chunk.tolist()]),
            (True, [(vector_chunk / np.linalg.norm(vector_chunk)).tolist()]),
        ]

        for (normalize_embeddings, expected_embeddings) in test_cases:
            with self.subTest(msg=f'Normalize embeddings: {normalize_embeddings}'):
                subfield1 = TensorField(
                    doc_id='id',
                    field_name='field_1',
                    modality=Modality.TEXT,
                    field_content="hello world!",
                    is_top_level_tensor_field=False,
                    is_multimodal_subfield=True
                )
                multimodal_field = MultiModalTensorField(
                    doc_id='id',
                    field_name='combo1',
                    weights={'field_1': 1.0, 'field_2': 2.0},
                    field_content='',
                    field_type=FieldType.MultimodalCombination,
                    subfields={'field_1': subfield1},
                    is_top_level_tensor_field=True,
                    normalize_embeddings=normalize_embeddings
                )

                subfield1.populate_chunks_and_embeddings(["hello world!"], [[1.0, 2.0]],
                                                         for_top_level_field=False)

                self.assertEqual(['{"field_1": "hello world!"}'], multimodal_field.tensor_field_chunks)
                self.assertEqual(expected_embeddings, multimodal_field.tensor_field_embeddings)

    def test_multimodal_combo_field_derived_from_multiple_subfields(self):
        # the final embedding should be weighted average if not normalised
        # field_1: [1.0, 2.0]; field_2: [2.0, 3.0], weight: [0.3, 0.7]
        vector_chunk = np.array([(1.0 * 0.3 + 2.0 * 0.7) / 2, (2.0 * 0.3 + 3.0 * 0.7) / 2])
        test_cases = [
            (False, [vector_chunk.tolist()]),
            (True, [(vector_chunk / np.linalg.norm(vector_chunk)).tolist()]),
        ]

        for (normalize_embeddings, expected_embeddings) in test_cases:
            with self.subTest(msg=f'Normalize embeddings: {normalize_embeddings}'):
                subfield1 = TensorField(
                    doc_id='id',
                    field_name='field_1',
                    modality=Modality.TEXT,
                    field_content="hello world!",
                    is_top_level_tensor_field=False,
                    is_multimodal_subfield=True
                )
                subfield2 = TensorField(
                    doc_id='id',
                    field_name='field_2',
                    modality=Modality.TEXT,
                    field_content="Hola!",
                    is_top_level_tensor_field=False,
                    is_multimodal_subfield=True
                )
                multimodal_field = MultiModalTensorField(
                    doc_id='id',
                    field_name='combo',
                    weights={'field_1': 0.3, 'field_2': 0.7},
                    field_content='',
                    field_type=FieldType.MultimodalCombination,
                    subfields={'field_1': subfield1, 'field_2': subfield2},
                    is_top_level_tensor_field=True,
                    normalize_embeddings=normalize_embeddings
                )

                subfield1.populate_chunks_and_embeddings(["hello world!"], [[1.0, 2.0]],
                                                         for_top_level_field=False)
                subfield2.populate_chunks_and_embeddings(["Hola!"], [[2.0, 3.0]],
                                                         for_top_level_field=False)

                self.assertEqual(['{"field_1": "hello world!", "field_2": "Hola!"}'],
                                 multimodal_field.tensor_field_chunks)
                self.assertEqual(expected_embeddings, multimodal_field.tensor_field_embeddings)

    def test_multimodal_combo_field_uses_pre_populated_chunks_and_embeddings(self):
        subfield1 = TensorField(
            doc_id='id',
            field_name='field_1',
            modality=Modality.TEXT,
            field_content="hello world!",
            is_top_level_tensor_field=False,
            is_multimodal_subfield=True
        )
        multimodal_field = MultiModalTensorField(
            doc_id='id',
            field_name='combo1',
            weights={'field_1': 1.0, 'field_2': 2.0},
            field_content='',
            field_type=FieldType.MultimodalCombination,
            subfields={'field_1': subfield1},
            is_top_level_tensor_field=True,
            normalize_embeddings=True
        )

        multimodal_field.populate_chunks_and_embeddings(["hello world!"], [[1.0, 2.0]])

        self.assertEqual(["hello world!"], multimodal_field.tensor_field_chunks)
        self.assertEqual([[1.0, 2.0]], multimodal_field.tensor_field_embeddings)


class TestTensorFieldPopulateChunksAndEmbeddings(unittest.TestCase):
    def test_populating_invalid_chunks_and_embeddings(self):
        field = TensorField(
            doc_id='id',
            field_name='field_1',
            modality=Modality.TEXT,
            field_content="hello world!",
            is_top_level_tensor_field=True,
            is_multimodal_subfield=False
        )

        valid_chunk = ['hello']
        valid_embeddings = [[1.0, 2.0]]
        test_cases = [
            # (chunks, embeddings)
            (None, valid_embeddings),
            ([], valid_embeddings),
            ('hello', valid_embeddings),
            ([1.0], valid_embeddings),

            (valid_chunk, None),
            (valid_chunk, []),
            (valid_chunk, 'hello'),
            (valid_chunk, ['hello']),
            (valid_chunk, [1.0, 2.0]),
            (valid_chunk, [['hello']]),
        ]
        for chunks, embeddings in test_cases:
            with self.subTest(chunks=chunks, embeddings=embeddings):
                with self.assertRaises(ValueError) as context:
                    field.populate_chunks_and_embeddings(chunks, embeddings)
                self.assertEqual(
                    'Invalid chunks and embeddings for doc: id, field: field_1',
                    str(context.exception))

    def test_populate_chunks_and_embeddings_raises_value_error_when_size_does_not_match(self):
        field = TensorField(
            doc_id='id',
            field_name='field_1',
            modality=Modality.TEXT,
            field_content="hello world!",
            is_top_level_tensor_field=True,
            is_multimodal_subfield=False
        )

        with self.assertRaises(ValueError) as context:
            field.populate_chunks_and_embeddings(["hello world!"], [[1.0, 2.0], [2.0, 3.0]])
        self.assertEqual('Chunk and embedding size does not match for doc: id, field: field_1: chunk size: 1, embedding size: 2',
                         str(context.exception))

    def test_populate_chunks_and_embeddings_for_top_level_field_without_modality(self):
        field = TensorField(
            doc_id='id',
            field_name='field_1',
            field_content="some long text",
            is_top_level_tensor_field=True,
            is_multimodal_subfield=False
        )

        field.populate_chunks_and_embeddings(["chunk1", "chunk2"], [[1.0, 2.0], [2.0, 3.0]])

        self.assertEqual(['chunk1', 'chunk2'], field.chunks)
        self.assertEqual([[1.0, 2.0], [2.0, 3.0]], field.embeddings)
        self.assertTrue(field.is_resolved)   # resolved since it's not a multimodal subfield

    def test_populate_chunks_and_embeddings_for_audio_video_field(self):
        """
        Test populating chunks and embeddings for audio video field.
        It always populates
        """

        test_cases = [
            # (is_top_level_field, is_sub_field, for_top_level_field)
            (True, False, True),  # top level field only, populated as top level field
            (False, True, False),  # subfield only, populated as subfield
            (True, True, True),  # both top level and subfield, populated as top level field
        ]

        for modality in [Modality.AUDIO, Modality.VIDEO]:
            for is_top_level_field, is_sub_field, for_top_level_field in test_cases:
                with self.subTest(modality=modality, is_top_level_field=is_top_level_field,
                                  is_sub_field=is_sub_field, for_top_level_field=for_top_level_field):
                    field = TensorField(
                        doc_id='id',
                        field_name='field_1',
                        field_content="some long text",
                        modality=modality,
                        is_top_level_tensor_field=is_top_level_field,
                        is_multimodal_subfield=is_sub_field
                    )

                    field.populate_chunks_and_embeddings(["chunk1", "chunk2"],
                                                         [[1.0, 2.0], [2.0, 3.0]],
                                                         for_top_level_field=for_top_level_field)

                    self.assertEqual(['chunk1', 'chunk2'], field.chunks)
                    self.assertEqual([[1.0, 2.0], [2.0, 3.0]], field.embeddings)
                    self.assertTrue(field.is_resolved)

    def test_populate_chunks_and_embeddings_for_image_text_subfield_only_should_raise_error_when_chunked(self):
        for modality in [Modality.TEXT, Modality.IMAGE]:
            with self.subTest(modality=modality):
                field = TensorField(
                    doc_id='id',
                    field_name='field_1',
                    field_content="text or url",
                    modality=modality,
                    is_top_level_tensor_field=False,
                    is_multimodal_subfield=True
                )
                with self.assertRaises(ValueError) as context:
                    field.populate_chunks_and_embeddings(["chunk1", "chunk2"],
                                                         [[1.0, 2.0], [2.0, 3.0]],
                                                         for_top_level_field=False)
                self.assertIn('field_1 of doc: id is a subfield and should not be chunked',
                              str(context.exception))

    def test_populate_chunks_and_embeddings_for_image_text_subfield_only_should_success_when_not_chunked(self):
        for modality in [Modality.TEXT, Modality.IMAGE]:
            with self.subTest(modality=modality):
                field = TensorField(
                    doc_id='id',
                    field_name='field_1',
                    field_content="text or url",
                    modality=modality,
                    is_top_level_tensor_field=False,
                    is_multimodal_subfield=True
                )
                field.populate_chunks_and_embeddings(["text or url"],
                                                     [[1.0, 2.0]],
                                                     for_top_level_field=False)
                self.assertEqual("text or url", field.subfield_chunk)
                self.assertEqual([1.0, 2.0], field.subfield_embedding)
                self.assertTrue(field.is_resolved)

    def test_populate_chunks_and_embeddings_for_image_text_both_top_level_and_sub_fields_not_chunked(self):
        for modality in [Modality.TEXT, Modality.IMAGE]:
            with self.subTest(modality=modality):
                field = TensorField(
                    doc_id='id',
                    field_name='field_1',
                    field_content="text or url",
                    modality=modality,
                    is_top_level_tensor_field=True,
                    is_multimodal_subfield=True
                )
                field.populate_chunks_and_embeddings(["text or url"],
                                                     [[1.0, 2.0]],
                                                     for_top_level_field=True)
                self.assertEqual(["text or url"], field.tensor_field_chunks)
                self.assertEqual([[1.0, 2.0]], field.tensor_field_embeddings)

                # as an optimisation, we also populate the subfield chunk and embeddings if not chunked
                self.assertEqual("text or url", field.subfield_chunk)
                self.assertEqual([1.0, 2.0], field.subfield_embedding)
                self.assertTrue(field.is_resolved)

    def test_populate_chunks_and_embeddings_for_image_text_both_top_level_and_sub_fields_chunked(self):
        for modality in [Modality.TEXT, Modality.IMAGE]:
            with self.subTest(modality=modality):
                field = TensorField(
                    doc_id='id',
                    field_name='field_1',
                    field_content="text or url",
                    modality=modality,
                    is_top_level_tensor_field=True,
                    is_multimodal_subfield=True
                )
                field.populate_chunks_and_embeddings(["chunk1", "chunk2"],
                                                     [[1.0, 2.0], [2.0, 3.0]],
                                                     for_top_level_field=True)
                self.assertEqual(["chunk1", "chunk2"], field.tensor_field_chunks)
                self.assertEqual([[1.0, 2.0], [2.0, 3.0]], field.tensor_field_embeddings)

                # we don't populate the subfield embedding if chunked embeddings are populated
                self.assertIsNone(field.multimodal_subfield_embedding)
                # not resolved since we still need to generate unchunked embeddings
                self.assertFalse(field.is_resolved)

    def test_populate_chunks_and_embeddings_for_text_image_both_top_level_and_sub_fields_with_single_different_chunk(self):
        """
        Tests when text and image (both subfield and top level field) gets populated as top level field with single
        chunk, but the chunk is different from the content. we should not populate the subfield embeddings since the
        chunk used for generating embedding is different
        """
        for modality in [Modality.TEXT, Modality.IMAGE]:
            with self.subTest(modality=modality):
                field = TensorField(
                    doc_id='id',
                    field_name='field_1',
                    field_content="text or url",
                    modality=modality,
                    is_top_level_tensor_field=True,
                    is_multimodal_subfield=True
                )
                field.populate_chunks_and_embeddings(["chunk1"],
                                                     [[1.0, 2.0]],
                                                     for_top_level_field=True)
                self.assertEqual(["chunk1"], field.tensor_field_chunks)
                self.assertEqual([[1.0, 2.0]], field.tensor_field_embeddings)

                # we don't populate the subfield embedding if the single chunk does not match the content
                self.assertIsNone(field.multimodal_subfield_embedding)
                # not resolved since we still need to generate unchunked embeddings
                self.assertFalse(field.is_resolved)
