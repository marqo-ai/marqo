import unittest
from typing import Dict, Any, List
from unittest.mock import patch

import pytest

from marqo.core.constants import MARQO_DOC_ID
from marqo.core.exceptions import DuplicateDocumentError, AddDocumentsError, MarqoDocumentParsingError, \
    InternalError
from marqo.core.inference.tensor_fields_container import TensorFieldsContainer
from marqo.core.models.add_docs_params import AddDocsParams, BatchVectorisationMode
from marqo.core.models.marqo_add_documents_response import MarqoAddDocumentsItem
from marqo.core.models.marqo_index import FieldType
from marqo.core.unstructured_vespa_index.unstructured_add_document_handler import \
    UnstructuredAddDocumentsHandler
from marqo.core.vespa_index.add_documents_handler import AddDocumentsResponseCollector, AddDocumentsHandler
from marqo.s2_inference import s2_inference
from marqo.s2_inference.errors import S2InferenceError
from marqo.s2_inference.types import Modality
from marqo.vespa.models import VespaDocument, FeedBatchResponse, FeedBatchDocumentResponse
from marqo.vespa.models.get_document_response import Document, GetBatchResponse, GetBatchDocumentResponse
from tests.marqo_test import MarqoTestCase
from tests.marqo_test import TestAudioUrls, TestVideoUrls, TestImageUrls


@pytest.mark.unittest
class TestAddDocumentHandler(MarqoTestCase):
    class DummyAddDocumentsHandler(AddDocumentsHandler):
        """
        We create a dummy implementation of the AddDocumentsHandler to verify the main workflow
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
            self.tensor_fields_container.collect(doc_id, field_name, field_content, self._infer_field_type)
            self.handled_fields.append((doc_id, field_name))

        def _handle_multi_modal_fields(self, marqo_doc: Dict[str, Any]) -> None:
            doc_id = marqo_doc[MARQO_DOC_ID]
            self.handled_multimodal_fields.append(doc_id)

        def _populate_existing_tensors(self, existing_vespa_docs: List[Document]) -> None:
            self.existing_vespa_docs = existing_vespa_docs

        def _to_vespa_doc(self, marqo_doc: Dict[str, Any]) -> VespaDocument:
            self.to_vespa_doc_call_count += 1
            return VespaDocument(id=marqo_doc[MARQO_DOC_ID], fields={})

        def _infer_field_type(self, field_name: str, field_content: Any) -> FieldType:
            return FieldType.Text

    @patch('marqo.vespa.vespa_client.VespaClient.feed_batch')
    @patch('marqo.vespa.vespa_client.VespaClient.get_batch')
    def test_add_documents_main_workflow_happy_path(self, mock_get_batch, mock_feed_batch):
        mock_get_batch.side_effect = [GetBatchResponse(errors=True, responses=[
            GetBatchDocumentResponse(id='id:index1:index1::1', pathId='path_id1',
                                     document=Document(id='id:index1:index1:1', fields={'marqo__id': '1'}), status=200),
            GetBatchDocumentResponse(id='id:index1:index1::2', pathId='path_id2', status=404),
            GetBatchDocumentResponse(id='id:index1:index1::3', pathId='path_id3', status=404)
        ])]
        mock_feed_batch.side_effect = [FeedBatchResponse(errors=False, responses=[
            FeedBatchDocumentResponse(id='id:index1:index1::1', pathId='path_id1', status=200),
            FeedBatchDocumentResponse(id='id:index1:index1::2', pathId='path_id2', status=200),
            FeedBatchDocumentResponse(id='id:index1:index1::3', pathId='path_id3', status=200),
        ])]

        handler = self.DummyAddDocumentsHandler(
            vespa_client=self.vespa_client,
            marqo_index=self.unstructured_marqo_index('index1', 'index1'),
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

    @patch('marqo.vespa.vespa_client.VespaClient.feed_batch')
    def test_add_documents_should_skip_duplicate_documents(self, mock_feed_batch):
        mock_feed_batch.side_effect = [FeedBatchResponse(errors=False, responses=[
            FeedBatchDocumentResponse(id='id:index1:index1::1', pathId='path_id1', status=200),
        ])]
        handler = self.DummyAddDocumentsHandler(
            vespa_client=self.vespa_client,
            marqo_index=self.unstructured_marqo_index('index1', 'index1'),
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

    @patch('marqo.vespa.vespa_client.VespaClient.feed_batch')
    def test_add_documents_should_skip_duplicate_documents_even_when_the_latter_one_errors_out(self, mock_feed_batch):
        handler = self.DummyAddDocumentsHandler(
            vespa_client=self.vespa_client,
            marqo_index=self.unstructured_marqo_index('index1', 'index1'),
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

        self.assertEqual(1, mock_feed_batch.call_count)
        self.assertEqual(([], 'index1'), mock_feed_batch.call_args_list[0][0])  # no vespa docs to persist

    @patch('marqo.vespa.vespa_client.VespaClient.feed_batch')
    def test_add_documents_should_handle_various_errors(self, mock_feed_batch):
        mock_feed_batch.side_effect = [FeedBatchResponse(errors=False, responses=[
            FeedBatchDocumentResponse(id='id:index1:index1::1', pathId='path_id1', status=400, message=
            'Could not parse field field1'),
            FeedBatchDocumentResponse(id='id:index1:index1::2', pathId='path_id2', status=429, message='vespa error2'),
            FeedBatchDocumentResponse(id='id:index1:index1::3', pathId='path_id3', status=507, message='vespa error3'),
        ])]

        handler = self.DummyAddDocumentsHandler(
            vespa_client=self.vespa_client,
            marqo_index=self.unstructured_marqo_index('index1', 'index1'),
            add_docs_params=AddDocsParams(
                index_name='index1', tensor_fields=['field1'],
                docs=[
                    {'_id': '1', 'field1': 'hello', 'field2': 2.0, 'field3': {'a': 1.0}},
                    {'_id': '2', 'field1': 'hello again'},
                    {'_id': '3', 'field1': 'hello world'},
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
                                  message='Marqo vector store receives too many requests. Please try again later',
                                  error='Marqo vector store receives too many requests. Please try again later',
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

    @patch('marqo.vespa.vespa_client.VespaClient.feed_batch')
    @patch('marqo.s2_inference.s2_inference.vectorise', wraps=s2_inference.vectorise)
    def test_add_documents_should_vectorise_tensor_fields_using_different_strategies(self, mock_vectorise, _):
        for batch_mode, expected_vectorise_call_count, expected_call_args in [
            (BatchVectorisationMode.PER_FIELD, 3, [['hello'], ['hello world'], ['ok']]),
            (BatchVectorisationMode.PER_DOCUMENT, 2, [['hello'], ['hello world', 'ok']]),
            (BatchVectorisationMode.PER_BATCH, 1, [['hello world', 'ok', 'hello']]),
        ]:
            with self.subTest(batch_mode=batch_mode):
                handler = self.DummyAddDocumentsHandler(
                    vespa_client=self.vespa_client,
                    marqo_index=self.unstructured_marqo_index('index1', 'index1'),
                    add_docs_params=AddDocsParams(
                        index_name='index1', tensor_fields=['field1', 'field4'],
                        batch_vectorisation_mode=batch_mode,
                        docs=[
                            {'_id': '1', 'field1': 'hello', 'field2': 2.0, 'field3': {'a': 1.0}},
                            {'_id': '2', 'field1': 'hello world', 'field4': 'ok'},
                        ])
                )

                mock_vectorise.reset_mock()

                handler.add_documents()
                self.assertEqual(expected_vectorise_call_count, mock_vectorise.call_count)
                # please note that assertCountEqual compares two list ignoring order
                self.assertCountEqual(expected_call_args,
                                      [args.kwargs['content'] for args in mock_vectorise.call_args_list])

    @patch('marqo.vespa.vespa_client.VespaClient.feed_batch')
    @patch('marqo.s2_inference.s2_inference.vectorise')
    def test_add_documents_should_fail_a_doc_using_vectorise_per_field_strategy(self, mock_vectorise, mock_feed_batch):
        mock_vectorise.side_effect = [S2InferenceError('vectorise error'), [[1.0, 2.0]]]
        mock_feed_batch.side_effect = [FeedBatchResponse(errors=False, responses=[
            FeedBatchDocumentResponse(id='id:index1:index1::1', pathId='path_id1', status=200),
        ])]
        handler = self.DummyAddDocumentsHandler(
            vespa_client=self.vespa_client,
            marqo_index=self.unstructured_marqo_index('index1', 'index1'),
            add_docs_params=AddDocsParams(
                index_name='index1', tensor_fields=['field1', 'field4'],
                batch_vectorisation_mode=BatchVectorisationMode.PER_FIELD,
                docs=[
                    {'_id': '1', 'field1': 'hello', 'field2': 2.0, 'field3': {'a': 1.0}},
                    {'_id': '2', 'field1': 'hello world', 'field4': 'ok'},
                ])
        )

        response = handler.add_documents()
        self.assertEqual(2, mock_vectorise.call_count)
        self.assertTrue(response.errors)
        self.assertEqual(200, response.items[0].status)
        self.assertEqual(400, response.items[1].status)
        self.assertEqual('vectorise error', response.items[1].message)

    @patch('marqo.vespa.vespa_client.VespaClient.feed_batch')
    @patch('marqo.s2_inference.s2_inference.vectorise')
    def test_add_documents_should_fail_a_doc_using_vectorise_per_doc_strategy(self, mock_vectorise, mock_feed_batch):
        mock_vectorise.side_effect = [S2InferenceError('vectorise error'), [[1.0, 2.0]]]
        mock_feed_batch.side_effect = [FeedBatchResponse(errors=False, responses=[
            FeedBatchDocumentResponse(id='id:index1:index1::1', pathId='path_id1', status=200),
        ])]
        handler = self.DummyAddDocumentsHandler(
            vespa_client=self.vespa_client,
            marqo_index=self.unstructured_marqo_index('index1', 'index1'),
            add_docs_params=AddDocsParams(
                index_name='index1', tensor_fields=['field1', 'field4'],
                batch_vectorisation_mode=BatchVectorisationMode.PER_DOCUMENT,
                docs=[
                    {'_id': '1', 'field1': 'hello', 'field2': 2.0, 'field3': {'a': 1.0}},
                    {'_id': '2', 'field1': 'hello world', 'field4': 'ok'},
                ])
        )

        response = handler.add_documents()
        self.assertEqual(2, mock_vectorise.call_count)
        self.assertTrue(response.errors)
        self.assertEqual(200, response.items[0].status)
        self.assertEqual(400, response.items[1].status)
        self.assertEqual('vectorise error', response.items[1].message)

    @patch('marqo.s2_inference.s2_inference.vectorise')
    def test_add_documents_should_fail_a_batch_using_vectorise_per_doc_strategy(self, mock_vectorise):
        mock_vectorise.side_effect = [S2InferenceError('vectorise error')]

        handler = self.DummyAddDocumentsHandler(
            vespa_client=self.vespa_client,
            marqo_index=self.unstructured_marqo_index('index1', 'index1'),
            add_docs_params=AddDocsParams(
                index_name='index1', tensor_fields=['field1', 'field4'],
                batch_vectorisation_mode=BatchVectorisationMode.PER_BATCH,
                docs=[
                    {'_id': '1', 'field1': 'hello', 'field2': 2.0, 'field3': {'a': 1.0}},
                    {'_id': '2', 'field1': 'hello world', 'field4': 'ok'},
                ])
        )

        with self.assertRaisesStrict(InternalError) as context:
            handler.add_documents()

        self.assertEqual('Encountered problem when vectorising batch of documents. Reason: vectorise error',
                         str(context.exception))

    def test_unstructured_add_documents_handler_infer_modality_logic_image_false_and_media_false(self):
        """Test the logic of the infer_modality method in UnstructuredAddDocumentsHandler when
        both treat_urls_and_pointers_as_images and treat_urls_and_pointers_as_media are False."""
        unstructured_add_documents_handler = UnstructuredAddDocumentsHandler(
            marqo_index=self.unstructured_marqo_index(
                'index1', 'index1',
                treat_urls_and_pointers_as_images=False,
                treat_urls_and_pointers_as_media=False
            ),
            add_docs_params=AddDocsParams(
                index_name='index1', tensor_fields=[], docs=[{'_id': '1'}]
            ),
            vespa_client=self.vespa_client
        )
        test_cases = [
            (TestAudioUrls.AUDIO1.value, "audio url should be treated as text"),
            (TestVideoUrls.VIDEO1.value, "video url should be treated as text"),
            (TestImageUrls.IMAGE1.value, "image url should be treated as text"),
        ]
        for url, test_case in test_cases:
            with self.subTest(msg=test_case):
                with (patch(
                        "marqo.core.unstructured_vespa_index.unstructured_add_document_handler.infer_modality") as
                      mock_infer_modality):
                    self.assertEqual(
                        FieldType.Text, unstructured_add_documents_handler.
                        _infer_field_type(field_name="dummy_field_name", field_content=url)
                    )
                mock_infer_modality.assert_not_called()

    def test_unstructured_add_documents_handler_infer_modality_logic_image_true_and_media_false(self):
        """Test the logic of the infer_modality method in UnstructuredAddDocumentsHandler when
        treat_urls_and_pointers_as_images=True and treat_urls_and_pointers_as_media=False."""
        unstructured_add_documents_handler = UnstructuredAddDocumentsHandler(
            marqo_index=self.unstructured_marqo_index(
                'index1', 'index1',
                treat_urls_and_pointers_as_images=True,
                treat_urls_and_pointers_as_media=False
            ),
            add_docs_params=AddDocsParams(
                index_name='index1', tensor_fields=[], docs=[{'_id': '1'}]
            ),
            vespa_client=self.vespa_client
        )
        test_cases = [
            (TestAudioUrls.AUDIO1.value, "audio url should be treated as text", FieldType.Text),
            (TestVideoUrls.VIDEO1.value, "video url should be treated as text", FieldType.Text),
            (TestImageUrls.IMAGE1.value, "image url should be treated as image", FieldType.ImagePointer),
        ]

        for url, test_case, expected_field_type in test_cases:
            with self.subTest(msg=test_case):
                self.assertEqual(
                    expected_field_type,
                    unstructured_add_documents_handler._infer_field_type(
                        field_name="dummy_field_name",
                        field_content=url
                    )
                )

    def test_unstructured_add_documents_handler_infer_modality_logic_image_true_and_media_true(self):
        """Test the logic of the infer_modality method in UnstructuredAddDocumentsHandler when
        treat_urls_and_pointers_as_images=True and treat_urls_and_pointers_as_media=True."""
        unstructured_add_documents_handler = UnstructuredAddDocumentsHandler(
            marqo_index=self.unstructured_marqo_index(
                'index1', 'index1',
                treat_urls_and_pointers_as_images=True,
                treat_urls_and_pointers_as_media=True
            ),
            add_docs_params=AddDocsParams(
                index_name='index1', tensor_fields=[], docs=[{'_id': '1'}]
            ),
            vespa_client=self.vespa_client
        )
        test_cases = [
            (TestAudioUrls.AUDIO1.value, "audio url should be treated as audio", FieldType.AudioPointer),
            (TestVideoUrls.VIDEO1.value, "video url should be treated as video", FieldType.VideoPointer),
            (TestImageUrls.IMAGE1.value, "image url should be treated as image", FieldType.ImagePointer),
        ]

        for url, test_case, expected_field_type in test_cases:
            with self.subTest(msg=test_case):
                self.assertEqual(
                    expected_field_type,
                    unstructured_add_documents_handler._infer_field_type(
                        field_name="dummy_field_name",
                        field_content=url
                    )
                )

    def test_collect_tensor_field_content_infer_modality_logic(self):
        """A test to ensure collect_tensor_field_content method in UnstructuredAddDocumentsHandler infer modality
        for tensor fields and multimodal sub-fields, but not for non-tensor fields."""
        unstructured_add_documents_handler = UnstructuredAddDocumentsHandler(
            marqo_index=self.unstructured_marqo_index(
                'index1', 'index1',
                treat_urls_and_pointers_as_images=True,
                treat_urls_and_pointers_as_media=True
            ),
            add_docs_params=AddDocsParams(
                index_name='index1', tensor_fields=["tensor_field", "my_multimodal_field"], docs=[{'_id': '1'}],
                mappings={
                    "my_multimodal_field":
                        {
                            "type": "multimodal_combination",
                            "weights": {
                                "text_field": 0.5, "image_field": 0.8
                            }
                        }
                }
            ),
            vespa_client=self.vespa_client
        )
        test_doc = {
            "non_tensor_field": TestAudioUrls.AUDIO1.value,
            "tensor_field": "This is a tensor field so its modality should be inferred",
            "text_field": "This is a sub field of my_multimodal_field so its modality should be inferred",
            "image_field": TestImageUrls.IMAGE1.value,
            "another_non_tensor_field": TestImageUrls.IMAGE1.value,
            "_id": "test"
        }

        test_cases = (
            ("non_tensor_field", "A non-tensor field should not be inferred for modality", False),
            ("tensor_field", "A tensor field should be inferred for modality", True),
            ("text_field", "A sub field of a multimodal field should be inferred for modality", True),
            ("image_field", "A sub field of a multimodal field should be inferred for modality", True),
            ("another_non_tensor_field", "A non-tensor field should not be inferred for modality", False),

        )

        for field_name, msg, called in test_cases:
            with self.subTest(f"{field_name} - {msg}"):
                with (patch("marqo.core.unstructured_vespa_index.unstructured_add_document_handler.infer_modality",
                            return_value=Modality.TEXT) as mock_infer_modality):
                    _ = unstructured_add_documents_handler._handle_field(
                        test_doc, field_name=field_name,
                        field_content=test_doc[field_name]
                    )
                if called:
                    mock_infer_modality.assert_called_once_with(test_doc[field_name], None)
                else:
                    mock_infer_modality.assert_not_called()


@pytest.mark.unittest
class TestAddDocumentsResponseCollector(unittest.TestCase):

    def test_should_collect_marqo_docs(self):
        collector = AddDocumentsResponseCollector()
        marqo_doc1 = {'_id': 'doc_id1'}
        marqo_doc2 = {'_id': 'doc_id2'}

        collector.collect_marqo_doc(1, marqo_doc1, 'doc_id1')
        collector.collect_marqo_doc(2, marqo_doc2, None)

        self.assertEqual(marqo_doc1, collector.marqo_docs['doc_id1'])
        self.assertEqual(marqo_doc2, collector.marqo_docs['doc_id2'])
        self.assertEqual(1, collector.marqo_doc_loc_map['doc_id1'])
        self.assertEqual(2, collector.marqo_doc_loc_map['doc_id2'])
        self.assertTrue(collector.visited('doc_id1'))
        self.assertFalse(collector.visited('doc_id2'))
        self.assertEqual({'doc_id1'}, collector.valid_original_ids())

    def test_collect_error_response_should_skip_duplicate_document_error(self):
        collector = AddDocumentsResponseCollector()
        collector.collect_error_response('doc_id1', DuplicateDocumentError('duplicate'))
        self.assertFalse(collector.errors)
        self.assertEqual([], collector.responses)

    def test_collect_error_response_should_capture_add_document_error_with_default_values(self):
        collector = AddDocumentsResponseCollector()
        collector.collect_error_response('doc_id1', AddDocumentsError('error message'), loc=1)
        self.assertTrue(collector.errors)
        loc, add_doc_item = collector.responses[0]
        self.assertEqual(1, loc)
        self.assertEqual('doc_id1', add_doc_item.id)
        self.assertEqual('error message', add_doc_item.message)
        self.assertEqual('error message', add_doc_item.error)
        self.assertEqual(400, add_doc_item.status)
        self.assertEqual('invalid_argument', add_doc_item.code)

    def test_collect_error_response_should_capture_add_document_error_with_custom_values(self):
        collector = AddDocumentsResponseCollector()
        collector.collect_error_response('doc_id1', AddDocumentsError('error message 2',
                                                                      error_code='err_code',
                                                                      status_code=403), loc=1)
        self.assertTrue(collector.errors)
        loc, add_doc_item = collector.responses[0]
        self.assertEqual(1, loc)
        self.assertEqual('doc_id1', add_doc_item.id)
        self.assertEqual('error message 2', add_doc_item.message)
        self.assertEqual('error message 2', add_doc_item.error)
        self.assertEqual(403, add_doc_item.status)
        self.assertEqual('err_code', add_doc_item.code)

    def test_collect_error_response_should_infer_loc_if_not_provided(self):
        collector = AddDocumentsResponseCollector()
        collector.collect_marqo_doc(5, {'_id': 'doc_id1'}, 'doc_id1')
        collector.collect_error_response('doc_id1', AddDocumentsError('error message'))
        loc, _ = collector.responses[0]
        self.assertEqual(5, loc)

    def test_collect_marqo_error_response_should_set_loc_to_none_if_not_provided(self):
        collector = AddDocumentsResponseCollector()
        collector.collect_error_response('doc_id1', AddDocumentsError('error message'))
        loc, _ = collector.responses[0]
        self.assertEqual(None, loc)

    def test_collect_marqo_error_response_should_remove_the_collected_marqo_doc(self):
        collector = AddDocumentsResponseCollector()
        collector.collect_marqo_doc(5, {'_id': 'doc_id1'}, 'doc_id1')
        self.assertIn('doc_id1', collector.marqo_docs)

        collector.collect_error_response('doc_id1', AddDocumentsError('error message'))
        self.assertNotIn('doc_id1', collector.marqo_docs)

    def test_collect_marqo_error_response_should_set_loc_to_none_if_doc_id_is_not_available(self):
        """
        This is possible due to persisting doc to Vespa do not always return doc_id when error is thrown.
        """
        collector = AddDocumentsResponseCollector()
        collector.collect_error_response(None, AddDocumentsError('error message'))
        loc, _ = collector.responses[0]
        self.assertEqual(None, loc)

    def test_collect_marqo_error_response_should_set_id_as_empty_if_original_id_is_none(self):
        """
        If _id is not provided in the request, we will generate a random one. And this information should not be
        returned to customer if this doc is not persisted. So we set the id in the error response to empty string
        """
        collector = AddDocumentsResponseCollector()
        collector.collect_marqo_doc(5, {'_id': 'doc_id1'}, None)
        collector.collect_error_response('doc_id1', AddDocumentsError('error message'))
        _, add_document_item = collector.responses[0]
        self.assertEqual('', add_document_item.id)

    def test_collect_marqo_error_response_should_set_doc_visited_if_original_id_is_present(self):
        """
        When dealing with duplicates, we only consider the last doc with that id, even it's not valid
        """
        collector = AddDocumentsResponseCollector()
        collector.collect_marqo_doc(5, {'_id': 'doc_id1'}, 'doc_id1')
        collector.collect_error_response('doc_id1', AddDocumentsError('error message'))
        self.assertTrue(collector.visited('doc_id1'))

    def test_collect_successful_response_should_add_200_as_status_code(self):
        collector = AddDocumentsResponseCollector()
        collector.collect_marqo_doc(5, {'_id': 'doc_id1'}, 'doc_id1')
        collector.collect_successful_response('doc_id1')
        loc, add_doc_item = collector.responses[0]
        self.assertEqual(5, loc)
        self.assertEqual('doc_id1', add_doc_item.id)
        self.assertEqual(200, add_doc_item.status)
        self.assertIsNone(add_doc_item.error)
        self.assertIsNone(add_doc_item.message)
        self.assertFalse(collector.errors)

    @patch('marqo.core.vespa_index.add_documents_handler.timer')
    def test_collect_final_responses(self, mock_timer):
        mock_timer.side_effect = [1.0, 2.0]
        collector = AddDocumentsResponseCollector()
        collector.collect_marqo_doc(1, {'_id': 'doc_id1'}, 'doc_id1')
        collector.collect_marqo_doc(2, {'_id': 'gen_doc_id2'}, None)
        collector.collect_marqo_doc(3, {'_id': 'doc_id3'}, None)
        collector.collect_error_response('doc_id4', AddDocumentsError('error message 4'), loc=4)
        collector.collect_error_response(None, AddDocumentsError('error message 1'))
        collector.collect_error_response('gen_doc_id2', AddDocumentsError('error message 2'))
        collector.collect_successful_response('doc_id3')

        # location should be reversed again in the response to revert the operation when we handle the batch of docs
        response = collector.to_add_doc_responses(index_name='index')
        self.assertTrue(response.errors)
        self.assertEqual('index', response.index_name)
        self.assertEqual(1000, response.processingTimeMs)

        self.assertEqual(4, len(response.items))
        self.assertEqual('doc_id4', response.items[0].id)  # doc_id4 is the original doc_id
        self.assertEqual('error message 4', response.items[0].message)
        self.assertEqual('doc_id3', response.items[1].id)  # doc_id3 should be returned since it's persisted
        self.assertEqual('',
                         response.items[2].id)  # gen_doc_id2 is generated, should not be returned for error
        self.assertEqual('error message 2', response.items[2].message)
        self.assertEqual('',
                         response.items[3].id)  # doc_id1 error message does not contain id, this came last
        self.assertEqual('error message 1', response.items[3].message)