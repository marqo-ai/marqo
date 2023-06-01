import unittest
from unittest.mock import patch

from marqo.tensor_search.models.add_docs_objects import AddDocsParams, Document

class TestDocument(unittest.TestCase):
    def test_get_or_generate_id_when_id_exists(self):
        document = Document(_id='existing_id')

        result = document.get_or_generate_id()

        self.assertEqual(result, 'existing_id')

    @patch('uuid.uuid4')
    def test_get_or_generate_id_when_id_does_not_exist(self, mock_uuid):
        mock_uuid.return_value = 'generated_id'
        document = Document()

        result = document.get_or_generate_id()

        self.assertEqual(result, 'generated_id')
        self.assertEqual(document._id, 'generated_id')

    @patch('marqo.tensor_search.validation.validate_id')
    def test_get_or_generate_id_when_id_gen_provides_invalid_id(self, mock_validate_id):
        mock_validate_id.side_effect = [ValueError('Invalid ID'), ValueError('Invalid ID')]

        document = Document()
        with self.assertRaises(ValueError):
            document.get_or_generate_id()

        document = Document()
        with self.assertRaises(ValueError):
            document.get_or_generate_id(id_gen=lambda: 'invalid_id')
            mock_validate_id.assert_called_once_with('invalid_id')


class TestAddDocsParams(unittest.TestCase):
    def test_create_anew_without_new_kwargs(self):
        params = AddDocsParams(index_name='index1', auto_refresh=True, docs=[Document(_id='doc')])

        result = params.create_anew()

        self.assertIsInstance(result, AddDocsParams)
        self.assertEqual(result.index_name, 'index1')
        self.assertEqual(result.auto_refresh, True)

    def test_create_anew_with_new_kwargs(self):
        params = AddDocsParams(index_name='index1', auto_refresh=True, docs=[Document(_id='doc')])
        new_docs = [Document(_id='doc1'), Document(_id='doc2')]

        result = params.create_anew(docs=new_docs, auto_refresh=False)

        self.assertIsInstance(result, AddDocsParams)
        self.assertEqual(result.index_name, 'index1')
        self.assertEqual(result.auto_refresh, False)
        self.assertEqual(result.docs, new_docs)