from unittest.mock import patch

import vespa.application as pyvespa

from marqo.vespa import concurrency
from marqo.vespa.models import VespaDocument
from marqo.vespa.models.feed_response import FeedBatchResponse
from marqo.vespa.vespa_client import VespaClient
from tests.marqo_test import AsyncMarqoTestCase


class TestFeedDocumentAsync(AsyncMarqoTestCase):
    TEST_SCHEMA = "test_vespa_client"
    TEST_CLUSTER = "content_default"

    def setUp(self):
        self.client = VespaClient("http://localhost:8080", "http://localhost:8080", "http://localhost:8080")
        self.pyvespa_client = pyvespa.Vespa(url="http://localhost", port=8080)

        self.pyvespa_client.delete_all_docs(self.TEST_CLUSTER, self.TEST_SCHEMA)

    def base_test_feed_batch_successful(self, func, batch):
        batch_ids = [doc.id for doc in batch]

        batch_response = func(batch, self.TEST_SCHEMA)

        self.assertIsInstance(batch_response, FeedBatchResponse)
        self.assertEqual(batch_response.errors, False)

        statuses = [response.status for response in batch_response.responses]
        pathIds = [response.pathId.split("/")[-1] for response in batch_response.responses]
        ids = [response.id.split("::")[-1] for response in batch_response.responses]
        messages = [response.message for response in batch_response.responses]

        self.assertEqual(statuses, ["200"] * len(batch))
        self.assertEqual(pathIds, batch_ids)
        self.assertEqual(ids, batch_ids)
        self.assertEqual(messages, [None] * len(batch))

    def test_feed_batch_successful(self):
        documents = [
            VespaDocument(id="doc1", fields={"title": "Title 1", "contents": "Content 1"}),
            VespaDocument(id="doc2", fields={"title": "Title 2"}),
        ]

        self.base_test_feed_batch_successful(self.client.feed_batch, documents)

    def test_feed_batch_emptyBatch_successful(self):
        documents = []

        self.base_test_feed_batch_successful(self.client.feed_batch, documents)

    @patch.object(concurrency, "_run_coroutine_in_thread", wraps=concurrency._run_coroutine_in_thread)
    async def test_feed_batch_existingEventLoop_successful(self, mock_executor):
        """Test that feed_batch works when an event loop is already running and runs in a new thread"""

        batch_response = self.client.feed_batch(
            [VespaDocument(id="doc1", fields={"title": "Title 1", "contents": "Content 1"})],
            self.TEST_SCHEMA
        )
        self.assertEqual(len(batch_response.responses), 1)

        mock_executor.assert_called_once()

    def test_feed_batch_noEventLoop_successful(self):
        """Test that feed_batch works when no event loop is running and doesn't use a new thread"""

        def raise_exception(*args, **kwargs):
            raise Exception("Attempted to run in new thread!")

        @patch.object(concurrency, "_run_coroutine_in_thread", side_effect=raise_exception)
        def run(mock_executor):
            batch_response = self.client.feed_batch(
                [VespaDocument(id="doc1", fields={"title": "Title 1", "contents": "Content 1"})],
                self.TEST_SCHEMA
            )
            self.assertEqual(len(batch_response.responses), 1)

        run()

    def test_feed_batch_sync_successful(self):
        documents = [
            VespaDocument(id="doc1", fields={"title": "Title 1", "contents": "Content 1"}),
            VespaDocument(id="doc2", fields={"title": "Title 2"}),
        ]

        self.base_test_feed_batch_successful(self.client.feed_batch_sync, documents)

    def test_feed_batch_sync_emptyBatch_successful(self):
        documents = []

        self.base_test_feed_batch_successful(self.client.feed_batch_sync, documents)

    def test_feed_batch_multithreaded_successful(self):
        documents = [
            VespaDocument(id="doc1", fields={"title": "Title 1", "contents": "Content 1"}),
            VespaDocument(id="doc2", fields={"title": "Title 2"}),
        ]

        self.base_test_feed_batch_successful(self.client.feed_batch_multithreaded, documents)

    def test_feed_batch_multithreaded_emptyBatch_successful(self):
        documents = []

        self.base_test_feed_batch_successful(self.client.feed_batch_multithreaded, documents)
