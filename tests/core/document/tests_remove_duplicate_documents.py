from marqo.core.models.marqo_index import *
from tests.marqo_test import MarqoTestCase


class TestUpdate(MarqoTestCase):

    def setUp(self) -> None:
        self.remove_duplicated_documents = self.config.document.remove_duplicated_documents

    def _base_test_delete_duplicate_documents(self, input_documents: List[Dict], expected_output: List[Dict]):
        output_documents = self.config.document.remove_duplicated_documents(input_documents)
        self.assertEqual(expected_output, output_documents)

    def test_normal_case(self):
        input_documents = [
            {'_id': '1', 'content': 'doc1'},
            {'_id': '2', 'content': 'doc2'}
        ]
        expected_output = (input_documents, {'1', '2'})
        self.assertEqual(self.remove_duplicated_documents(input_documents), expected_output)

    def test_duplicates(self):
        input_documents = [
            {'_id': '1', 'content': 'doc1'},
            {'_id': '1', 'content': 'doc1 updated'},
            {'_id': '2', 'content': 'doc2'}
        ]
        expected_output = ([{'_id': '1', 'content': 'doc1 updated'}, {'_id': '2', 'content': 'doc2'}], {'1', '2'})
        self.assertEqual(self.remove_duplicated_documents(input_documents), expected_output)

    def test_non_dict_entries(self):
        input_documents = [
            "not_a_doc",
            {'_id': '2', 'content': 'doc2'}
        ]
        expected_output = (
        ["not_a_doc", {'_id': '2', 'content': 'doc2'}], {'2'})  # Expecting non-dict entries to be included
        self.assertEqual(self.remove_duplicated_documents(input_documents), expected_output)

    def test_missing_ids(self):
        input_documents = [
            {'content': 'doc1'},
            {'_id': '2', 'content': 'doc2'}
        ]
        expected_output = (input_documents, {'2'})  # Documents without an `_id` are retained
        self.assertEqual(self.remove_duplicated_documents(input_documents), expected_output)

    def test_invalid_ids(self):
        """Non-hashable ids are still retained"""
        input_documents = [
            {'_id': ['invalid'], 'content': 'doc1'},
            {'_id': '2', 'content': 'doc2'}
        ]
        expected_output = (input_documents, {'2'})
        self.assertEqual(self.remove_duplicated_documents(input_documents), expected_output)

    def test_empty_list(self):
        input_documents = []
        expected_output = ([], set())
        self.assertEqual(self.remove_duplicated_documents(input_documents), expected_output)

    def test_all_duplicates_except_one(self):
        input_documents = [
            {'_id': '1', 'content': 'doc1'},
            {'_id': '1', 'content': 'doc1 updated'},
            {'_id': '1', 'content': 'doc1 final'}
        ]
        expected_output = ([{'_id': '1', 'content': 'doc1 final'}], {'1'})
        self.assertEqual(self.remove_duplicated_documents(input_documents), expected_output)

    def test_mixed_types_of_ids(self):
        input_documents = [
            {'_id': '1', 'content': 'doc1'},
            {'_id': 2, 'content': 'doc2'},
            {'_id': 2.5, 'content': 'doc3'}
        ]
        expected_output = (input_documents, {'1', 2, 2.5})
        self.assertEqual(self.remove_duplicated_documents(input_documents), expected_output)

