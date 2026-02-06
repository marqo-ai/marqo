from unittest.mock import Mock, MagicMock, call, patch

from marqo.core.document.document import Document
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.vespa.exceptions import VespaNotConvergedError
from tests.unit_tests.marqo_test import MarqoTestCase


class TestDocumentConvergence(MarqoTestCase):

    def setUp(self):
        self.mock_vespa_client = Mock()
        self.mock_index_management = Mock()
        self.mock_inference = Mock()

    def test_add_documents_waits_for_convergence_before_get_index(self):
        """Verify wait_for_application_convergence is called before get_index."""
        call_order = []

        self.mock_vespa_client.wait_for_application_convergence.side_effect = (
            lambda **kwargs: call_order.append('wait_for_convergence')
        )
        self.mock_index_management.get_index.side_effect = (
            lambda name: call_order.append('get_index') or self._make_structured_index(name)
        )

        # Mock the handler that would be created
        with patch('marqo.core.document.document.StructuredAddDocumentsHandler') as mock_handler_cls:
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

            self.assertEqual(call_order, ['wait_for_convergence', 'get_index'])

    def test_add_documents_convergence_failure_propagates(self):
        """Verify convergence failure propagates and get_index is never called."""
        self.mock_vespa_client.wait_for_application_convergence.side_effect = (
            VespaNotConvergedError("Vespa application did not converge within 120 seconds.")
        )

        doc = Document(self.mock_vespa_client, self.mock_index_management, self.mock_inference)
        add_docs_params = AddDocsParams(
            index_name="test_index",
            docs=[{"_id": "doc1", "title": "test"}],
            tensor_fields=[],
        )

        with self.assertRaises(VespaNotConvergedError):
            doc.add_documents(add_docs_params)

        self.mock_index_management.get_index.assert_not_called()

    def test_add_documents_convergence_uses_configured_timeout(self):
        """Verify custom timeout is passed to wait_for_application_convergence."""
        self.mock_index_management.get_index.return_value = self._make_structured_index("test_index")

        with patch('marqo.core.document.document.StructuredAddDocumentsHandler') as mock_handler_cls:
            mock_handler = Mock()
            mock_handler.add_documents.return_value = Mock()
            mock_handler_cls.return_value = mock_handler

            custom_timeout = 30
            doc = Document(self.mock_vespa_client, self.mock_index_management, self.mock_inference,
                           convergence_timeout_seconds=custom_timeout)
            add_docs_params = AddDocsParams(
                index_name="test_index",
                docs=[{"_id": "doc1", "title": "test"}],
                tensor_fields=[],
            )

            doc.add_documents(add_docs_params)

            self.mock_vespa_client.wait_for_application_convergence.assert_called_once_with(
                timeout=custom_timeout
            )

    def test_add_documents_passes_convergence_timeout_to_handler(self):
        """Verify convergence timeout is passed to the handler constructor."""
        self.mock_index_management.get_index.return_value = self._make_structured_index("test_index")

        with patch('marqo.core.document.document.StructuredAddDocumentsHandler') as mock_handler_cls:
            mock_handler = Mock()
            mock_handler.add_documents.return_value = Mock()
            mock_handler_cls.return_value = mock_handler

            custom_timeout = 45
            doc = Document(self.mock_vespa_client, self.mock_index_management, self.mock_inference,
                           convergence_timeout_seconds=custom_timeout)
            add_docs_params = AddDocsParams(
                index_name="test_index",
                docs=[{"_id": "doc1", "title": "test"}],
                tensor_fields=[],
            )

            doc.add_documents(add_docs_params)

            _, kwargs = mock_handler_cls.call_args
            self.assertEqual(kwargs['convergence_timeout_seconds'], custom_timeout)

    def _make_structured_index(self, name):
        from marqo.core.models.marqo_index import StructuredMarqoIndex, Model, TextPreProcessing, TextSplitMethod, \
            ImagePreProcessing, DistanceMetric, VectorNumericType, HnswConfig
        from marqo.version import get_version
        return StructuredMarqoIndex(
            name=name,
            schema_name=name,
            model=Model(name='hf/all_datasets_v4_MiniLM-L6'),
            normalize_embeddings=True,
            text_preprocessing=TextPreProcessing(split_length=2, split_overlap=0, split_method=TextSplitMethod.Sentence),
            image_preprocessing=ImagePreProcessing(patch_method=None),
            distance_metric=DistanceMetric.Angular,
            vector_numeric_type=VectorNumericType.Float,
            hnsw_config=HnswConfig(ef_construction=128, m=16),
            fields=[],
            tensor_fields=[],
            marqo_version=get_version(),
            created_at=1,
            updated_at=2,
            version=None,
        )
