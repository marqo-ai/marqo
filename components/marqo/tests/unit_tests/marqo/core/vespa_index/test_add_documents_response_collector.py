import unittest
from unittest.mock import patch

from marqo.core.exceptions import DuplicateDocumentError, AddDocumentsError
from marqo.core.vespa_index.add_documents_handler import AddDocumentsResponseCollector


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