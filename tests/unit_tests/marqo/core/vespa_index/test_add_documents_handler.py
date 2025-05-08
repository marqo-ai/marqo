from typing import Dict, Any, List
from unittest.mock import MagicMock

import numpy as np

from marqo.core.constants import MARQO_DOC_ID
from marqo.core.exceptions import AddDocumentsError, MarqoDocumentParsingError
from marqo.core.inference.api import Inference, InferenceRequest, InferenceResult, Modality, TextChunkConfig, \
    ChunkConfig
from marqo.core.inference.tensor_fields_container import TensorFieldsContainer, TensorField
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_add_documents_response import MarqoAddDocumentsItem
from marqo.core.models.marqo_index import TextPreProcessing, TextSplitMethod, ImagePreProcessing, PatchMethod, \
    AudioPreProcessing, VideoPreProcessing, Model
from marqo.core.vespa_index.add_documents_handler import AddDocumentsHandler
from marqo.vespa.models import VespaDocument, FeedBatchResponse, FeedBatchDocumentResponse
from marqo.vespa.models.get_document_response import Document, GetBatchResponse, GetBatchDocumentResponse
from marqo.vespa.vespa_client import VespaClient
from unit_tests.marqo_test import MarqoTestCase
from integ_tests.marqo_test import MarqoTestCase as fixture


class DummyAddDocumentsHandler(AddDocumentsHandler):
    """
    We create a dummy stub of the AddDocumentsHandler to verify the main workflow
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.handled_fields = []
        self.handled_multimodal_fields = []
        self.existing_vespa_docs = []
        self.to_vespa_doc_call_count = 0

    def _create_tensor_fields_container(self) -> TensorFieldsContainer:
        return TensorFieldsContainer(self.add_docs_params.tensor_fields, [], {}, True)

    def _handle_field(self, marqo_doc, field_name, field_content) -> None:
        doc_id = marqo_doc[MARQO_DOC_ID]
        marqo_doc[field_name] = field_content
        self.tensor_fields_container.collect(doc_id, field_name, field_content)
        self.handled_fields.append((doc_id, field_name))

    def _handle_multi_modal_fields(self, marqo_doc: Dict[str, Any]) -> None:
        doc_id = marqo_doc[MARQO_DOC_ID]
        self.handled_multimodal_fields.append(doc_id)

    def _populate_existing_tensors(self, existing_vespa_docs: List[Document]) -> None:
        self.existing_vespa_docs = existing_vespa_docs

    def _to_vespa_doc(self, marqo_doc: Dict[str, Any]) -> VespaDocument:
        self.to_vespa_doc_call_count += 1
        return VespaDocument(id=marqo_doc[MARQO_DOC_ID], fields={})

    def _infer_modality(self, tensor_field: TensorField) -> Modality:
        return Modality.TEXT


class TestAddDocumentHandler(MarqoTestCase):

    def setUp(self):
        self.vespa_client = MagicMock(spec=VespaClient)
        self.vespa_client.translate_vespa_document_response.side_effect = VespaClient.translate_vespa_document_response
        self.inference = MagicMock(spec=Inference)

        def vectorise_side_effect(request: InferenceRequest) -> InferenceResult:
            return InferenceResult(result=[[('chunk', np.array([1.0, 2.0]))] for _ in request.contents])

        self.inference.vectorise.side_effect = vectorise_side_effect

    def test_add_documents_main_workflow_happy_path(self):
        self.vespa_client.get_batch.side_effect = [GetBatchResponse(errors=True, responses=[
            GetBatchDocumentResponse(id='id:index1:index1::1', pathId='path_id1',
                                     document=Document(id='id:index1:index1:1', fields={'marqo__id': '1'}), status=200),
            GetBatchDocumentResponse(id='id:index1:index1::2', pathId='path_id2', status=404),
            GetBatchDocumentResponse(id='id:index1:index1::3', pathId='path_id3', status=404)
        ])]
        self.vespa_client.feed_batch.side_effect = [FeedBatchResponse(errors=False, responses=[
            FeedBatchDocumentResponse(id='id:index1:index1::1', pathId='path_id1', status=200),
            FeedBatchDocumentResponse(id='id:index1:index1::2', pathId='path_id2', status=200),
            FeedBatchDocumentResponse(id='id:index1:index1::3', pathId='path_id3', status=200),
        ])]

        handler = DummyAddDocumentsHandler(
            vespa_client=self.vespa_client,
            inference=self.inference,
            marqo_index=fixture.unstructured_marqo_index('index1', 'index1'),
            add_docs_params=AddDocsParams(
                index_name='index1',
                tensor_fields=['field1'],
                use_existing_tensors=True,
                docs=[
                    {'_id': '1', 'field1': 'hello', 'field2': 2.0, 'field3': {'a': 1.0}},
                    {'_id': '2', 'field1': 'hello again', 'field2': 3.0, 'field4': ['abcd']},
                    {'_id': '3', 'field2': ['de'], 'field5': {'content': 'a', 'vector': [0.1] * 32}},
                ])
        )

        response = handler.add_documents()

        self.assertFalse(response.errors)
        self.assertEqual('index1', response.index_name)
        self.assertEqual(3, len(response.items))
        for i in range(3):
            self.assertEqual(str(i + 1), response.items[i].id)
            self.assertEqual(200, response.items[i].status)

        # verify the workflow call the abstract methods
        self.assertEqual({
            ('1', 'field1'), ('1', 'field2'), ('1', 'field3'),
            ('2', 'field1'), ('2', 'field2'), ('2', 'field4'),
            ('3', 'field2'), ('3', 'field5')
        }, set(handler.handled_fields))

        self.assertEqual({'3', '2', '1'}, set(handler.handled_multimodal_fields))

        self.assertEqual([Document(id='id:index1:index1:1', fields={'marqo__id': '1'})],
                         handler.existing_vespa_docs)  # only the doc with 200 status code is passed to the method

        self.assertEqual(3, handler.to_vespa_doc_call_count)

    def test_add_documents_should_skip_duplicate_documents(self):
        self.vespa_client.feed_batch.side_effect = [FeedBatchResponse(errors=False, responses=[
            FeedBatchDocumentResponse(id='id:index1:index1::1', pathId='path_id1', status=200),
        ])]
        handler = DummyAddDocumentsHandler(
            vespa_client=self.vespa_client,
            inference=self.inference,
            marqo_index=fixture.unstructured_marqo_index('index1', 'index1'),
            add_docs_params=AddDocsParams(
                index_name='index1', tensor_fields=['field1'],
                docs=[
                    {'_id': '1', 'field1': 'hello', 'field2': 2.0, 'field3': {'a': 1.0}},
                    {'_id': '1', 'field4': ['de'], 'field5': {'content': 'a', 'vector': [0.1] * 32}},
                ])
        )

        self.assertFalse(handler.add_documents().errors)
        self.assertEqual({
            ('1', 'field4'), ('1', 'field5'),  # the second doc with the same id overrides the first one
        }, set(handler.handled_fields))
        self.assertEqual(1, handler.to_vespa_doc_call_count)

    def test_add_documents_should_skip_duplicate_documents_even_when_the_latter_one_errors_out(self):
        self.vespa_client.feed_batch.side_effect = [FeedBatchResponse(responses=[], errors=False)]

        handler = DummyAddDocumentsHandler(
            vespa_client=self.vespa_client,
            inference=self.inference,
            marqo_index=fixture.unstructured_marqo_index('index1', 'index1'),
            add_docs_params=AddDocsParams(
                index_name='index1', tensor_fields=['field1'],
                docs=[
                    {'_id': '1', 'field1': 'hello', 'field2': 2.0, 'field3': {'a': 1.0}},
                    {'_id': '1', 'field4': ['de'], 'field5': {'content': 'a', 'vector': [0.1] * 32}},
                ])
        )

        # override the handle field method to raise an error when handling field5
        def handle_field_raise_error(self, marqo_doc, field_name, _) -> None:
            if field_name == 'field5':
                raise AddDocumentsError('some error')
            self.handled_fields.append((marqo_doc[MARQO_DOC_ID], field_name))

        handler._handle_field = handle_field_raise_error.__get__(handler)

        response = handler.add_documents()
        self.assertTrue(response.errors)
        self.assertTrue(1, len(response.items))
        self.assertEqual('some error', response.items[0].message)

        self.assertEqual([('1', 'field4')], handler.handled_fields)
        self.assertEqual(0, handler.to_vespa_doc_call_count)

        self.assertEqual(1, self.vespa_client.feed_batch.call_count)
        self.assertEqual(([], 'index1'), self.vespa_client.feed_batch.call_args_list[0][0])  # no vespa docs to persist

    def test_add_documents_should_handle_various_errors(self):
        self.vespa_client.feed_batch.side_effect = [FeedBatchResponse(errors=False, responses=[
            FeedBatchDocumentResponse(id='id:index1:index1::1', pathId='path_id1', status=400, message='Could not parse field field1'),
            FeedBatchDocumentResponse(id='id:index1:index1::2', pathId='path_id2', status=429, message='vespa error2'),
            FeedBatchDocumentResponse(id='id:index1:index1::3', pathId='path_id3', status=507, message='vespa error3'),
        ])]

        handler = DummyAddDocumentsHandler(
            vespa_client=self.vespa_client,
            inference=self.inference,
            marqo_index=fixture.unstructured_marqo_index('index1', 'index1'),
            add_docs_params=AddDocsParams(
                index_name='index1', tensor_fields=['field1'],
                docs=[
                    {'_id': '1', 'field1': 'hello', 'field2': 2.0, 'field3': {'a': 1.0}},  # vespa 400
                    {'_id': '2', 'field1': 'hello again'},  # vespa 429
                    {'_id': '3', 'field1': 'hello world'},  # vespa 507
                    {'bad_field': 'bad_content'},  # error out when converting to vespa doc
                    {'_id': [5], 'field4': ['de']},  # doc with invalid id
                    {'field4': ['de'], 'field5': 'a very large string object' * 10000},  # doc too large
                    {},  # empty doc
                    [2.0] * 32  # doc is not a dict
                ])
        )

        def to_vespa_doc_throw_error(_, marqo_doc: Dict[str, Any]) -> VespaDocument:
            if marqo_doc.get('bad_field') == 'bad_content':
                raise MarqoDocumentParsingError('MarqoDocumentParsingError')
            return VespaDocument(id=marqo_doc[MARQO_DOC_ID], fields={})

        handler._to_vespa_doc = to_vespa_doc_throw_error.__get__(handler)

        response = handler.add_documents()
        self.assertTrue(response.errors)

        self.assertEqual([
            MarqoAddDocumentsItem(status=400, id='1',
                                  message='The document contains invalid characters in the fields. Original error: Could not parse field field1 ',
                                  error='The document contains invalid characters in the fields. Original error: Could not parse field field1 ',
                                  code='vespa_error'),
            MarqoAddDocumentsItem(status=429, id='2',
                                  message='Marqo vector store received too many requests. Please try again later',
                                  error='Marqo vector store received too many requests. Please try again later',
                                  code='vespa_error'),
            MarqoAddDocumentsItem(status=400, id='3', message='Marqo vector store is out of memory or disk space',
                                  error='Marqo vector store is out of memory or disk space', code='vespa_error'),
            MarqoAddDocumentsItem(status=400, id='', message='MarqoDocumentParsingError',
                                  error='MarqoDocumentParsingError', code='invalid_argument'),
            MarqoAddDocumentsItem(status=400, id='',
                                  message='Document _id must be a string type! Received _id [5] of type `list`',
                                  error='Document _id must be a string type! Received _id [5] of type `list`',
                                  code='invalid_document_id'),
            MarqoAddDocumentsItem(status=400, id='',
                                  message='Document with length `260032` exceeds the allowed document size limit of [100000].',
                                  error='Document with length `260032` exceeds the allowed document size limit of [100000].',
                                  code='doc_too_large'),
            MarqoAddDocumentsItem(status=400, id='', message="Can't index an empty dict.",
                                  error="Can't index an empty dict.", code='invalid_argument'),
            MarqoAddDocumentsItem(status=400, id='', message='Docs must be dicts',
                                  error='Docs must be dicts',
                                  code='invalid_argument')
        ], response.items)

    def test_preprocessing_config_for_text_modality(self):
        handler = DummyAddDocumentsHandler(
            vespa_client=self.vespa_client,
            inference=self.inference,
            marqo_index=fixture.unstructured_marqo_index(
                'index1', 'index1',
                text_preprocessing=TextPreProcessing(
                    split_length=100,
                    split_overlap=10,
                    split_method=TextSplitMethod.Word
                ),
                model=Model(
                    name='hf/all_datasets_v4_MiniLM-L6',
                    text_chunk_prefix='default_prefix:'
                )
            ),
            add_docs_params=AddDocsParams(
                index_name='index1', tensor_fields=['field1'],
                docs=[{'_id': '1', 'field1': 'hello'}],
                text_chunk_prefix='prefix1:'
            ),
        )

        for_top_level_field = handler._get_preprocessing_config(Modality.TEXT, for_top_level_field=True)
        self.assertEqual(for_top_level_field.text_prefix, 'prefix1:')
        self.assertTrue(for_top_level_field.should_chunk)
        self.assertEqual(for_top_level_field.chunk_config, TextChunkConfig(split_length=100, split_overlap=10, split_method='word'))

        for_subfield = handler._get_preprocessing_config(Modality.TEXT, for_top_level_field=False)
        self.assertEqual(for_subfield.text_prefix, 'prefix1:')
        self.assertFalse(for_subfield.should_chunk)
        self.assertIsNone(for_subfield.chunk_config)

    def test_preprocessing_config_for_text_modality_with_default_prefix(self):
        handler = DummyAddDocumentsHandler(
            vespa_client=self.vespa_client,
            inference=self.inference,
            marqo_index=fixture.unstructured_marqo_index(
                'index1', 'index1',
                text_preprocessing=TextPreProcessing(
                    split_length=100,
                    split_overlap=10,
                    split_method=TextSplitMethod.Word
                ),
                model=Model(
                    name='hf/all_datasets_v4_MiniLM-L6',
                    text_chunk_prefix='default_prefix:'
                )
            ),
            add_docs_params=AddDocsParams(
                index_name='index1', tensor_fields=['field1'],
                docs=[{'_id': '1', 'field1': 'hello'}],
            ),
        )

        for_top_level_field = handler._get_preprocessing_config(Modality.TEXT, for_top_level_field=True)
        self.assertEqual(for_top_level_field.text_prefix, 'default_prefix:')
        self.assertTrue(for_top_level_field.should_chunk)
        self.assertEqual(for_top_level_field.chunk_config, TextChunkConfig(split_length=100, split_overlap=10, split_method='word'))

        for_subfield = handler._get_preprocessing_config(Modality.TEXT, for_top_level_field=False)
        self.assertEqual(for_subfield.text_prefix, 'default_prefix:')
        self.assertFalse(for_subfield.should_chunk)
        self.assertIsNone(for_subfield.chunk_config)

    def test_preprocessing_config_for_image_modality_without_patch_method(self):
        handler = DummyAddDocumentsHandler(
            vespa_client=self.vespa_client,
            inference=self.inference,
            marqo_index=fixture.unstructured_marqo_index('index1', 'index1'),
            add_docs_params=AddDocsParams(
                index_name='index1', tensor_fields=['field1'],
                docs=[{'_id': '1', 'field1': 'hello'}],
                media_download_headers={'a': 'b'},
                image_download_thread_count=3
            ),
        )

        for_top_level_field = handler._get_preprocessing_config(Modality.IMAGE, for_top_level_field=True)
        self.assertFalse(for_top_level_field.should_chunk)
        self.assertIsNone(for_top_level_field.patch_method)
        self.assertEqual(for_top_level_field.download_header, {'a': 'b'})
        self.assertEqual(for_top_level_field.download_thread_count, 3)

        for_subfield = handler._get_preprocessing_config(Modality.IMAGE, for_top_level_field=False)
        self.assertFalse(for_subfield.should_chunk)
        self.assertIsNone(for_subfield.patch_method)
        self.assertEqual(for_subfield.download_header, {'a': 'b'})
        self.assertEqual(for_subfield.download_thread_count, 3)

    def test_preprocessing_config_for_image_modality_with_patch_method(self):
        handler = DummyAddDocumentsHandler(
            vespa_client=self.vespa_client,
            inference=self.inference,
            marqo_index=fixture.unstructured_marqo_index(
                'index1', 'index1',
                image_preprocessing=ImagePreProcessing(patch_method=PatchMethod.Simple)
            ),
            add_docs_params=AddDocsParams(
                index_name='index1', tensor_fields=['field1'],
                docs=[{'_id': '1', 'field1': 'hello'}],
                image_download_thread_count=5
            ),
        )

        for_top_level_field = handler._get_preprocessing_config(Modality.IMAGE, for_top_level_field=True)
        self.assertTrue(for_top_level_field.should_chunk)
        self.assertEqual(for_top_level_field.patch_method, 'simple')
        self.assertIsNone(for_top_level_field.download_header)
        self.assertEqual(for_top_level_field.download_thread_count, 5)

        for_subfield = handler._get_preprocessing_config(Modality.IMAGE, for_top_level_field=False)
        self.assertFalse(for_subfield.should_chunk)
        self.assertIsNone(for_subfield.patch_method)
        self.assertIsNone(for_subfield.download_header)
        self.assertEqual(for_subfield.download_thread_count, 5)

    def test_preprocessing_config_for_audio_modality(self):
        handler = DummyAddDocumentsHandler(
            vespa_client=self.vespa_client,
            inference=self.inference,
            marqo_index=fixture.unstructured_marqo_index(
                'index1', 'index1',
                audio_preprocessing=AudioPreProcessing(split_length=25, split_overlap=5)
            ),
            add_docs_params=AddDocsParams(
                index_name='index1', tensor_fields=['field1'],
                docs=[{'_id': '1', 'field1': 'hello'}],
                media_download_thread_count=4,
                media_download_headers={'a': 'b'},
            ),
        )

        for_top_level_field = handler._get_preprocessing_config(Modality.AUDIO, for_top_level_field=True)
        self.assertTrue(for_top_level_field.should_chunk)
        self.assertEqual(for_top_level_field.chunk_config, ChunkConfig(split_length=25, split_overlap=5))
        self.assertEqual(for_top_level_field.download_header, {'a': 'b'})
        self.assertEqual(for_top_level_field.download_thread_count, 4)

        for_subfield = handler._get_preprocessing_config(Modality.AUDIO, for_top_level_field=False)
        self.assertTrue(for_subfield.should_chunk)
        self.assertEqual(for_top_level_field.chunk_config, ChunkConfig(split_length=25, split_overlap=5))
        self.assertEqual(for_subfield.download_header, {'a': 'b'})
        self.assertEqual(for_subfield.download_thread_count, 4)

    def test_preprocessing_config_for_video_modality(self):
        handler = DummyAddDocumentsHandler(
            vespa_client=self.vespa_client,
            inference=self.inference,
            marqo_index=fixture.unstructured_marqo_index(
                'index1', 'index1',
                video_preprocessing=VideoPreProcessing(split_length=25, split_overlap=5)
            ),
            add_docs_params=AddDocsParams(
                index_name='index1', tensor_fields=['field1'],
                docs=[{'_id': '1', 'field1': 'hello'}],
                media_download_thread_count=6,
                media_download_headers={'a': 'b'},
            ),
        )

        for_top_level_field = handler._get_preprocessing_config(Modality.VIDEO, for_top_level_field=True)
        self.assertTrue(for_top_level_field.should_chunk)
        self.assertEqual(for_top_level_field.chunk_config, ChunkConfig(split_length=25, split_overlap=5))
        self.assertEqual(for_top_level_field.download_header, {'a': 'b'})
        self.assertEqual(for_top_level_field.download_thread_count, 6)

        for_subfield = handler._get_preprocessing_config(Modality.VIDEO, for_top_level_field=False)
        self.assertTrue(for_subfield.should_chunk)
        self.assertEqual(for_top_level_field.chunk_config, ChunkConfig(split_length=25, split_overlap=5))
        self.assertEqual(for_subfield.download_header, {'a': 'b'})
        self.assertEqual(for_subfield.download_thread_count, 6)