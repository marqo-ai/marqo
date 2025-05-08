from unittest.mock import patch

import httpx
import vespa.application as pyvespa

from marqo.vespa.exceptions import VespaError
from marqo.vespa.models import VespaDocument
from marqo.vespa.vespa_client import VespaClient
from integ_tests.marqo_test import AsyncMarqoTestCase


class TestFeedDocumentAsync(AsyncMarqoTestCase):
    TEST_SCHEMA = "test_vespa_client"
    TEST_CLUSTER = "content_default"

    def setUp(self):
        self.client = VespaClient("http://localhost:19071", "http://localhost:8080",
                                  "http://localhost:8080", "content_default")
        self.pyvespa_client = pyvespa.Vespa(url="http://localhost", port=8080)

        self.pyvespa_client.delete_all_docs(self.TEST_CLUSTER, self.TEST_SCHEMA)

    def _base_test_update_documents_batch_successful(self, func, batch):
        batch_ids = [doc.id for doc in batch]
        batch_response = func(batch, self.TEST_SCHEMA)

        self.assertEqual(batch_response.errors, False)

        statuses = [response.status for response in batch_response.responses]
        path_ids = [response.path_id.split("/")[-1] for response in batch_response.responses]
        ids = [response.id.split("::")[-1] for response in batch_response.responses]
        messages = [response.message for response in batch_response.responses]

        self.assertEqual(statuses, [200] * len(batch))
        self.assertEqual(path_ids, batch_ids)
        self.assertEqual(ids, batch_ids)
        self.assertEqual(messages, [None] * len(batch))

    def test_update_documents_batch_successful(self):
        original_documents = [
            VespaDocument(id="doc1", fields={"title": "Title 1", "contents": "Content 1", "marqo__id": "doc1"}),
            VespaDocument(id="doc2", fields={"title": "Title 2", "marqo__id": "doc2"}),
        ]
        # we need to feed the original documents first
        r = self.client.feed_batch(original_documents, self.TEST_SCHEMA)

        update_documents = [
            VespaDocument(id="doc1", fields={"title": {"assign": "Title 1 update"}}),
            VespaDocument(id="doc2", fields={"title": {"assign": "Title 2 updated"}}),
        ]

        self._base_test_update_documents_batch_successful(self.client.update_documents_batch, update_documents)

    def test_update_documents_batch_emptyBatch_successful(self):
        documents = []

        self._base_test_update_documents_batch_successful(self.client.feed_batch, documents)

    def test_feed_batch_documents_do_not_exists(self):
        update_documents = [
            VespaDocument(id="doc1", fields={"title": {"assign": "Title 1 update"}}),
            VespaDocument(id="doc2", fields={"title": {"assign": "Title 2 updated"}}),
        ]

        batch_response = self.client.update_documents_batch(update_documents, self.TEST_SCHEMA)

        statuses = [response.status for response in batch_response.responses]
        path_ids = [response.path_id.split("/")[-1] for response in batch_response.responses]
        ids = [response.id.split("::")[-1] for response in batch_response.responses if response.status == 200]
        messages = [response.message for response in batch_response.responses]

        self.assertEqual([404, 404], statuses)
        self.assertEqual(["doc1", "doc2"], path_ids)
        self.assertEqual([], ids)
        self.assertIn("not exist", messages[0])
        self.assertIn("not exist", messages[1])

    def test_feed_batch_documents_invalid_values(self):
        original_documents = [
            VespaDocument(id="doc1", fields={"title": "Title 1", "contents": "Content 1", "marqo__id": "doc1"}),
            VespaDocument(id="doc2", fields={"title": "Title 2", "marqo__id": "doc2"}),
        ]
        # we need to feed the original documents first
        r = self.client.feed_batch(original_documents, self.TEST_SCHEMA)

        update_documents = [
            VespaDocument(id="doc1", fields={"title": {"assign": "Title 1 update"}}),
            VespaDocument(id="doc2", fields={"title": {"assign": [1, 2, 3]}}), # Invalid list value for string field
        ]

        batch_response = self.client.update_documents_batch(update_documents, self.TEST_SCHEMA)

        statuses = [response.status for response in batch_response.responses]
        path_ids = [response.path_id.split("/")[-1] for response in batch_response.responses]
        ids = [response.id.split("::")[-1] for response in batch_response.responses if response.status == 200]
        messages = [response.message for response in batch_response.responses]

        self.assertEqual([200, 400], statuses)
        self.assertEqual(["doc1", "doc2"], path_ids)
        self.assertEqual(["doc1"], ids)
        self.assertIsNone(messages[0])
        self.assertIsNotNone(messages[1])

    def test_update_documents_batch_network_error(self):
        update_documents = [
            VespaDocument(id="doc1", fields={"title": {"assign": "Network Failure Test"}}),
            VespaDocument(id="doc2", fields={"title": {"assign": "Network Failure Test"}})
        ]
        with patch("httpx.AsyncClient.put", side_effect=httpx.NetworkError("Network failure")):
            batch_response = self.client.update_documents_batch(update_documents, self.TEST_SCHEMA)

        self.assertEqual(batch_response.errors, True)
        self.assertEqual(2, len(batch_response.responses))
        for r in batch_response.responses:
            self.assertEqual(r.status, 500)
            self.assertIn("Network Error", r.message)