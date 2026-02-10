import os
from unittest.mock import Mock, patch

from marqo.core.document.document import Document
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_index import Model
from marqo.vespa.exceptions import VespaNotConvergedError
from tests.unit_tests.marqo_test import MarqoTestCase


class TestDocumentConvergence(MarqoTestCase):

    def setUp(self):
        self.mock_vespa_client = Mock()
        self.mock_index_management = Mock()
        self.mock_inference = Mock()

    def test_add_documents_checks_convergence_after_get_index(self):
        """Verify check_for_application_convergence is called after get_index."""
        call_order = []

        self.mock_vespa_client.check_for_application_convergence.side_effect = (
            lambda: call_order.append('check_convergence')
        )
        self.mock_index_management.get_index.side_effect = (
            lambda name: call_order.append('get_index') or self._make_semi_structured_index(name)
        )

        # Mock the handler that would be created
        with patch('marqo.core.document.document.SemiStructuredAddDocumentsHandler') as mock_handler_cls:
            mock_handler = Mock()
            mock_handler.add_documents.return_value = Mock()
            mock_handler_cls.return_value = mock_handler

            doc = Document(self.mock_vespa_client, self.mock_index_management, self.mock_inference)
            add_docs_params = AddDocsParams(
                index_name="test_index",
                docs=[{"_id": "doc1", "title": "test"}],
                tensor_fields=[],
            )

            doc.add_documents(add_docs_params)

            self.assertEqual(call_order, ['get_index', 'check_convergence'])

    def test_add_documents_convergence_failure_propagates(self):
        """Verify convergence failure propagates after get_index is called."""
        self.mock_index_management.get_index.return_value = self._make_semi_structured_index("test_index")
        self.mock_vespa_client.check_for_application_convergence.side_effect = (
            VespaNotConvergedError("Vespa application has not converged.")
        )

        doc = Document(self.mock_vespa_client, self.mock_index_management, self.mock_inference)
        add_docs_params = AddDocsParams(
            index_name="test_index",
            docs=[{"_id": "doc1", "title": "test"}],
            tensor_fields=[],
        )

        with self.assertRaises(VespaNotConvergedError):
            doc.add_documents(add_docs_params)

        self.mock_index_management.get_index.assert_called_once_with("test_index")

    def test_add_documents_skips_convergence_when_env_var_disabled(self):
        """Verify convergence check is skipped when MARQO_ENABLE_ADD_DOCUMENTS_CONVERGENCE_CHECK=FALSE."""
        self.mock_index_management.get_index.return_value = self._make_semi_structured_index("test_index")

        with patch('marqo.core.document.document.SemiStructuredAddDocumentsHandler') as mock_handler_cls, \
                patch.dict(os.environ, {"MARQO_ENABLE_ADD_DOCUMENTS_CONVERGENCE_CHECK": "FALSE"}):
            mock_handler = Mock()
            mock_handler.add_documents.return_value = Mock()
            mock_handler_cls.return_value = mock_handler

            doc = Document(self.mock_vespa_client, self.mock_index_management, self.mock_inference)
            add_docs_params = AddDocsParams(
                index_name="test_index",
                docs=[{"_id": "doc1", "title": "test"}],
                tensor_fields=[],
            )

            doc.add_documents(add_docs_params)

            self.mock_vespa_client.check_for_application_convergence.assert_not_called()
            self.mock_index_management.get_index.assert_called_once_with("test_index")

    def _make_semi_structured_index(self, name):
        return self.semi_structured_marqo_index(
            name=name,
            model=Model(name='hf/all_datasets_v4_MiniLM-L6'),
        )