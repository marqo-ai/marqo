import unittest
from typing import List, Tuple, Union

import numpy as np
import pytest
from PIL.Image import Image

from marqo.core.inference.tensor_fields_container import TensorFieldContent, Chunker, MultiModalTensorFieldContent, \
    Vectoriser
from marqo.core.exceptions import AddDocumentsError
from marqo.core.models.marqo_index import FieldType


class DummyChunker(Chunker):
    def __init__(self, no_chunking=False):
        self.no_chunking = no_chunking

    def chunk(self, _: str, single_chunk: bool = False) -> Tuple[List[str], List[str]]:
        if single_chunk or self.no_chunking:
            return ['single_chunk'], ['single_content_chunk']
        else:
            return ['chunk1', 'chunk2'], ['content_chunk1', 'content_chunk2']


class DummyVectoriser(Vectoriser):
    def __init__(self):
        self.vectorise_call_count = 0

    def vectorise(self, content_chunks: Union[List[str], List[Image]]) -> List[List[float]]:
        self.vectorise_call_count += 1
        return [[1.0 * (i + 1), 2.0 * (i + 1)] for i in range(len(content_chunks))]


@pytest.mark.unittest
class TestTensorFieldContent(unittest.TestCase):

    def test_populate_chunks_and_embeddings(self):
        test_cases = [
            # (type, is_tensor_field, is_subfield, resolved_after_population)
            (FieldType.Text, True, False, True),
            (FieldType.Text, True, True, False),  # populate method won't be run for a non-toplevel tensor field
            (FieldType.ImagePointer, True, False, True),
            (FieldType.ImagePointer, True, True, False),
            (FieldType.AudioPointer, True, False, True),
            (FieldType.AudioPointer, True, True, True),  # audio subfield does not need to generate single chunk
            (FieldType.VideoPointer, True, False, True),
            (FieldType.VideoPointer, True, True, True),  # video subfield does not need to generate single chunk
        ]

        for (field_type, is_tensor_field, is_subfield, resolved_after_population) in test_cases:
            with self.subTest(msg=f'field of type {field_type} (toplevel: {is_tensor_field}, subfield: {is_subfield})'):
                tensor_field_content = TensorFieldContent(
                    field_type=field_type,
                    field_content="hello world!",
                    is_tensor_field=is_tensor_field,
                    is_multimodal_subfield=is_subfield
                )

                tensor_field_content.populate_chunks_and_embeddings(['hello world'], [[1.0, 1.2]])
                self.assertEquals(['hello world'], tensor_field_content.chunks)
                self.assertEquals([], tensor_field_content.content_chunks)
                self.assertEquals([[1.0, 1.2]], tensor_field_content.embeddings)
                self.assertEquals(resolved_after_population, tensor_field_content.is_resolved)

    def test_chunk_resolved_field_will_not_generate_more_content_chunks(self):
        tensor_field_content = TensorFieldContent(
            field_type=FieldType.Text,
            field_content="hello world!",
            is_tensor_field=True
        )
        tensor_field_content.populate_chunks_and_embeddings(['hello world'], [[1.0, 1.2]])
        self.assertTrue(tensor_field_content.is_resolved)
        tensor_field_content.chunk({FieldType.Text: DummyChunker(False)})
        self.assertEqual([], tensor_field_content.content_chunks)

    def test_chunk_populated_multimodal_subfield_will_generate_content_chunks(self):
        tensor_field_content = TensorFieldContent(
            field_type=FieldType.Text,
            field_content="hello world!",
            is_tensor_field=True,
            is_multimodal_subfield=True
        )
        tensor_field_content.populate_chunks_and_embeddings(['hello world'], [[1.0, 1.2]])
        self.assertFalse(tensor_field_content.is_resolved)
        tensor_field_content.chunk({FieldType.Text: DummyChunker(False)})
        self.assertEqual(['single_content_chunk'], tensor_field_content.content_chunks)

    def test_chunk_populated_multimodal_subfield_will_not_generate_content_chunks_if_chunk_is_same(self):
        tensor_field_content = TensorFieldContent(
            field_type=FieldType.Text,
            field_content="hello world!",
            is_tensor_field=True,
            is_multimodal_subfield=True
        )
        tensor_field_content.populate_chunks_and_embeddings(['single_chunk'], [[1.0, 1.2]])
        self.assertFalse(tensor_field_content.is_resolved)
        tensor_field_content.chunk({FieldType.Text: DummyChunker(False)})
        self.assertEqual([], tensor_field_content.content_chunks)

    def test_chunk_with_no_chunker_for_the_type(self):
        tensor_field_content = TensorFieldContent(
            field_type=FieldType.Text,
            field_content="hello world!",
            is_tensor_field=True
        )
        with self.assertRaises(AddDocumentsError) as e:
            tensor_field_content.chunk({FieldType.ImagePointer: DummyChunker(False)})
        self.assertIn('Chunking is not supported for field type: Text', str(e.exception))
        self.assertEqual([], tensor_field_content.chunks)
        self.assertEqual([], tensor_field_content.content_chunks)

    def test_chunk_top_level_tensor_field(self):
        tensor_field_content = TensorFieldContent(
            field_type=FieldType.Text,
            field_content="hello world!",
            is_tensor_field=True
        )
        tensor_field_content.chunk({FieldType.Text: DummyChunker(False)})
        self.assertEqual(['chunk1', 'chunk2'], tensor_field_content.chunks)
        self.assertEqual(['content_chunk1', 'content_chunk2'], tensor_field_content.content_chunks)

    def test_chunk_subfield(self):
        tensor_field_content = TensorFieldContent(
            field_type=FieldType.Text,
            field_content="hello world!",
            is_tensor_field=False,
            is_multimodal_subfield=True
        )
        tensor_field_content.chunk({FieldType.Text: DummyChunker(False)})
        self.assertEqual(['single_chunk'], tensor_field_content.chunks)
        self.assertEqual(['single_content_chunk'], tensor_field_content.content_chunks)

    def test_chunk_both_subfield_and_top_level_field(self):
        """
        When a field is both top-level tensor field and subfield, and chunking is enabled,
        It stores both the chunks for the tensor field and the single chunk for the subfield.
        In this way, the vectorisation will just need to be done once.
        """
        tensor_field_content = TensorFieldContent(
            field_type=FieldType.Text,
            field_content="hello world!",
            is_tensor_field=True,
            is_multimodal_subfield=True
        )
        tensor_field_content.chunk({FieldType.Text: DummyChunker(False)})
        self.assertEqual(['chunk1', 'chunk2', 'single_chunk'], tensor_field_content.chunks)
        self.assertEqual(['content_chunk1', 'content_chunk2', 'single_content_chunk'],
                         tensor_field_content.content_chunks)
        self.assertEqual(['chunk1', 'chunk2'], tensor_field_content.tensor_field_chunks)
        self.assertEqual('single_chunk', tensor_field_content.sub_field_chunk)

    def test_chunk_both_subfield_and_top_level_field_no_chunking(self):
        """
        Following last test case, if chunker does not do chunking (based on index config), the chunk
        generated should be the same. To avoid vectorising the same content twice, we just store the single
        chunk once
        """
        tensor_field_content = TensorFieldContent(
            field_type=FieldType.Text,
            field_content="hello world!",
            is_tensor_field=True,
            is_multimodal_subfield=True
        )
        tensor_field_content.chunk({FieldType.Text: DummyChunker(True)})
        self.assertEqual(['single_chunk'], tensor_field_content.chunks)
        self.assertEqual(['single_content_chunk'], tensor_field_content.content_chunks)
        self.assertEqual(['single_chunk'], tensor_field_content.tensor_field_chunks)
        self.assertEqual('single_chunk', tensor_field_content.sub_field_chunk)

    def test_chunk_subfield_for_populated_tensor_field(self):
        tensor_field_content = TensorFieldContent(
            field_type=FieldType.Text,
            field_content="hello world!",
            is_tensor_field=True,
            is_multimodal_subfield=True
        )
        tensor_field_content.populate_chunks_and_embeddings(['hello world'], [[1.0, 1.2]])
        tensor_field_content.chunk({FieldType.Text: DummyChunker(False)})
        self.assertEqual(['hello world', 'single_chunk'], tensor_field_content.chunks)
        self.assertEqual(['single_content_chunk'], tensor_field_content.content_chunks)
        self.assertEqual(['hello world'], tensor_field_content.tensor_field_chunks)
        self.assertEqual('single_chunk', tensor_field_content.sub_field_chunk)

    def test_chunk_audio_video_field(self):
        """
        Audio and Video fields are chunked regardless of whether they are top-level or subfields
        When they are subfields, the field content will be returned as the subfield chunk.
        """
        test_cases = [
            ('top-level audio field', FieldType.AudioPointer, True, False),
            ('subfield audio field', FieldType.AudioPointer, False, True),
            ('audio field both top-level and subfield', FieldType.AudioPointer, True, True),
            ('top-level video field', FieldType.VideoPointer, True, False),
            ('subfield video field', FieldType.VideoPointer, False, True),
            ('video field both top-level and subfield', FieldType.VideoPointer, True, True),
        ]

        for (test_case, field_type, is_tensor_field, is_multimodal_subfield) in test_cases:
            with self.subTest(msg=test_case):
                tensor_field_content = TensorFieldContent(
                    field_type=field_type,
                    field_content="URL",
                    is_tensor_field=is_tensor_field,
                    is_multimodal_subfield=is_multimodal_subfield
                )

                tensor_field_content.chunk({field_type: DummyChunker(False)})
                self.assertEqual(['chunk1', 'chunk2'], tensor_field_content.chunks)
                self.assertEqual(['content_chunk1', 'content_chunk2'], tensor_field_content.content_chunks)
                self.assertEqual(['chunk1', 'chunk2'], tensor_field_content.tensor_field_chunks)
                self.assertEqual('URL' if is_multimodal_subfield else None, tensor_field_content.sub_field_chunk)

    def test_vectorisation_without_vectoriser_for_the_type(self):
        tensor_field_content = TensorFieldContent(
            field_type=FieldType.Text,
            field_content="hello world!",
            is_tensor_field=True
        )
        with self.assertRaises(AddDocumentsError) as e:
            tensor_field_content.vectorise({})
        self.assertIn('Vectorisation is not supported for field type: Text', str(e.exception))
        self.assertEqual([], tensor_field_content.embeddings)

    def test_vectorisation_without_content_chunk(self):
        tensor_field_content = TensorFieldContent(
            field_type=FieldType.Text,
            field_content="hello world!",
            is_tensor_field=True
        )
        vectoriser = DummyVectoriser()
        tensor_field_content.vectorise({FieldType.Text: vectoriser})
        self.assertEqual([], tensor_field_content.embeddings)
        self.assertEqual(0, vectoriser.vectorise_call_count)
        self.assertEqual([], tensor_field_content.tensor_field_embeddings)
        self.assertEqual(None, tensor_field_content.sub_field_embedding)

    def test_vectorisation_with_content_chunks(self):
        tensor_field_content = TensorFieldContent(
            field_type=FieldType.Text,
            field_content="hello world!",
            is_tensor_field=True
        )
        tensor_field_content.chunk({FieldType.Text: DummyChunker(False)})
        vectoriser = DummyVectoriser()
        tensor_field_content.vectorise({FieldType.Text: vectoriser})
        self.assertEqual([[1.0, 2.0], [2.0, 4.0]], tensor_field_content.embeddings)
        self.assertEqual(1, vectoriser.vectorise_call_count)
        self.assertEqual([[1.0, 2.0], [2.0, 4.0]], tensor_field_content.tensor_field_embeddings)
        self.assertEqual(None, tensor_field_content.sub_field_embedding)

    def test_vectorisation_with_subfield_chunks(self):
        tensor_field_content = TensorFieldContent(
            field_type=FieldType.Text,
            field_content="hello world!",
            is_tensor_field=True,
            is_multimodal_subfield=True
        )
        tensor_field_content.chunk({FieldType.Text: DummyChunker(False)})
        vectoriser = DummyVectoriser()
        tensor_field_content.vectorise({FieldType.Text: vectoriser})
        # There are in total 3 chunks to vectorise
        self.assertEqual([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], tensor_field_content.embeddings)
        self.assertEqual(1, vectoriser.vectorise_call_count)
        self.assertEqual([[1.0, 2.0], [2.0, 4.0]], tensor_field_content.tensor_field_embeddings)
        self.assertEqual([3.0, 6.0], tensor_field_content.sub_field_embedding)

    def test_vectorisation_with_populated_tensor_fields_and_unpopulated_sub_fields(self):
        tensor_field_content = TensorFieldContent(
            field_type=FieldType.Text,
            field_content="hello world!",
            is_tensor_field=True,
            is_multimodal_subfield=True
        )
        tensor_field_content.populate_chunks_and_embeddings(['hello world'], [[1.1, 1.2]])
        tensor_field_content.chunk({FieldType.Text: DummyChunker(False)})
        vectoriser = DummyVectoriser()
        tensor_field_content.vectorise({FieldType.Text: vectoriser})
        # The first embeddings is the pre-populated value
        self.assertEqual([[1.1, 1.2], [1.0, 2.0]], tensor_field_content.embeddings)
        self.assertEqual(1, vectoriser.vectorise_call_count)
        self.assertEqual([[1.1, 1.2]], tensor_field_content.tensor_field_embeddings)
        self.assertEqual([1.0, 2.0], tensor_field_content.sub_field_embedding)

    def test_vectorise_audio_video_field(self):
        """
        Audio and Video fields are chunked regardless of whether they are top-level or subfields
        When they are subfields, the field content will be returned as the subfield chunk.
        """
        test_cases = [
            ('top-level audio field', FieldType.AudioPointer, True, False),
            ('subfield audio field', FieldType.AudioPointer, False, True),
            ('audio field both top-level and subfield', FieldType.AudioPointer, True, True),
            ('top-level video field', FieldType.VideoPointer, True, False),
            ('subfield video field', FieldType.VideoPointer, False, True),
            ('video field both top-level and subfield', FieldType.VideoPointer, True, True),
        ]

        for (test_case, field_type, is_tensor_field, is_multimodal_subfield) in test_cases:
            with self.subTest(msg=test_case):
                tensor_field_content = TensorFieldContent(
                    field_type=field_type,
                    field_content="URL",
                    is_tensor_field=is_tensor_field,
                    is_multimodal_subfield=is_multimodal_subfield
                )

                tensor_field_content.chunk({field_type: DummyChunker(False)})
                vectoriser = DummyVectoriser()
                tensor_field_content.vectorise({field_type: vectoriser})
                self.assertEqual([[1.0, 2.0], [2.0, 4.0]], tensor_field_content.embeddings)
                self.assertEqual(1, vectoriser.vectorise_call_count)
                self.assertEqual([[1.0, 2.0], [2.0, 4.0]], tensor_field_content.tensor_field_embeddings)
                if is_multimodal_subfield:
                    # average will be returned for multi-modal calculation
                    self.assertEqual([1.5, 3.0], tensor_field_content.sub_field_embedding)
                else:
                    self.assertEqual(None, tensor_field_content.sub_field_embedding)

    def test_multimodal_combo_field_populated_from_existing_field(self):
        subfield1 = TensorFieldContent(
            field_type=FieldType.Text,
            field_content="hello world!",
            is_tensor_field=False,
            is_multimodal_subfield=True
        )
        multimodal_field = MultiModalTensorFieldContent(
            weights={'field_1': 1.0, 'field_2': 2.0},
            field_content='',
            field_type=FieldType.MultimodalCombination,
            subfields={'field_1': subfield1},
            is_tensor_field=True,
            normalize_embeddings=True
        )

        multimodal_field.populate_chunks_and_embeddings(['{"field_1": "hello world"}'], [[1.1, 1.2]])
        self.assertEqual(['{"field_1": "hello world"}'], multimodal_field.tensor_field_chunks)
        self.assertEqual([[1.1, 1.2]], multimodal_field.tensor_field_embeddings)

    def test_multimodal_combo_field_with_no_subfield(self):
        multimodal_field = MultiModalTensorFieldContent(
            weights={'field_1': 1.0, 'field_2': 2.0},
            field_content='',
            field_type=FieldType.MultimodalCombination,
            subfields={},
            is_tensor_field=True,
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
                subfield1 = TensorFieldContent(
                    field_type=FieldType.Text,
                    field_content="hello world!",
                    is_tensor_field=False,
                    is_multimodal_subfield=True
                )
                multimodal_field = MultiModalTensorFieldContent(
                    weights={'field_1': 1.0, 'field_2': 2.0},
                    field_content='',
                    field_type=FieldType.MultimodalCombination,
                    subfields={'field_1': subfield1},
                    is_tensor_field=True,
                    normalize_embeddings=normalize_embeddings
                )

                subfield1.chunk({FieldType.Text: DummyChunker(False)})
                subfield1.vectorise({FieldType.Text: DummyVectoriser()})
                self.assertEqual(['{"field_1": "single_chunk"}'], multimodal_field.tensor_field_chunks)
                self.assertEqual(expected_embeddings, multimodal_field.tensor_field_embeddings)

    def test_multimodal_combo_field_derived_from_multiple_subfields(self):
        # the final embedding should be weighted average if not normalised
        vector_chunk = np.array([(1.0 * 1.0 + 1.0 * 2.0) / 2, (2.0 * 1.0 + 2.0 * 2.0) / 2])
        test_cases = [
            (False, [vector_chunk.tolist()]),
            (True, [(vector_chunk / np.linalg.norm(vector_chunk)).tolist()]),
        ]

        for (normalize_embeddings, expected_embeddings) in test_cases:
            with self.subTest(msg=f'Normalize embeddings: {normalize_embeddings}'):

                subfield1 = TensorFieldContent(
                    field_type=FieldType.Text,
                    field_content="hello world!",
                    is_tensor_field=False,
                    is_multimodal_subfield=True
                )
                subfield2 = TensorFieldContent(
                    field_type=FieldType.Text,
                    field_content="Hola!",
                    is_tensor_field=False,
                    is_multimodal_subfield=True
                )
                multimodal_field = MultiModalTensorFieldContent(
                    weights={'field_1': 1.0, 'field_2': 2.0},
                    field_content='',
                    field_type=FieldType.MultimodalCombination,
                    subfields={'field_1': subfield1, 'field_2': subfield2},
                    is_tensor_field=True,
                    normalize_embeddings=normalize_embeddings
                )

                chunkers = {FieldType.Text: DummyChunker(False)}
                vectorisers = {FieldType.Text: DummyVectoriser()}
                subfield1.chunk(chunkers)
                subfield1.vectorise(vectorisers)
                subfield2.chunk(chunkers)
                subfield2.vectorise(vectorisers)
                self.assertEqual(['{"field_1": "single_chunk", "field_2": "single_chunk"}'], multimodal_field.tensor_field_chunks)
                self.assertEqual(expected_embeddings, multimodal_field.tensor_field_embeddings)
