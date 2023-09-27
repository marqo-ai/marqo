import os
from unittest.mock import patch

import httpx
import vespa.application as pyvespa

from marqo.vespa import concurrency
from marqo.vespa.exceptions import VespaError
from marqo.vespa.models import VespaDocument
from marqo.vespa.vespa_client import VespaClient
from tests.marqo_test import AsyncMarqoTestCase


class TestFeedDocumentAsync(AsyncMarqoTestCase):
    TEST_SCHEMA = "test_vespa_client"
    TEST_CLUSTER = "content_default"

    def setUp(self):
        self.client = VespaClient("http://localhost:19071", "http://localhost:8080", "http://localhost:8080")
        self.pyvespa_client = pyvespa.Vespa(url="http://localhost", port=8080)

        self.pyvespa_client.delete_all_docs(self.TEST_CLUSTER, self.TEST_SCHEMA)

    def _base_test_feed_batch_successful(self, func, batch):
        batch_ids = [doc.id for doc in batch]

        batch_response = func(batch, self.TEST_SCHEMA)

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

        self._base_test_feed_batch_successful(self.client.feed_batch, documents)

    def test_feed_batch_emptyBatch_successful(self):
        documents = []

        self._base_test_feed_batch_successful(self.client.feed_batch, documents)

    def test_feed_batch_invalidDoc_successful(self):
        documents = [
            VespaDocument(id="doc1", fields={"title": "Title 1", "contents": "Content 1"}),
            VespaDocument(id="doc2", fields={"invalid_field": "Title 2"}),
        ]

        batch_response = self.client.feed_batch(documents, self.TEST_SCHEMA)

        self.assertEqual(batch_response.errors, True)

        statuses = [response.status for response in batch_response.responses]
        pathIds = [response.pathId.split("/")[-1] for response in batch_response.responses]
        ids = [response.id.split("::")[-1] for response in batch_response.responses if response.status == "200"]
        messages = [response.message for response in batch_response.responses]

        self.assertEqual(statuses, ["200", "400"])
        self.assertEqual(pathIds, ["doc1", "doc2"])
        self.assertEqual(ids, ["doc1"])
        self.assertIsNone(messages[0])
        self.assertIsNotNone(messages[1])

    def test_feed_batch_invalidFeedUrl_fails(self):
        feed_client = VespaClient("http://localhost:8080", "http://localhost:8000", "http://localhost:8080")
        documents = [
            VespaDocument(id="doc1", fields={"title": "Title 1", "contents": "Content 1"}),
            VespaDocument(id="doc2", fields={"title": "Title 2"}),
        ]

        with self.assertRaises(VespaError):
            feed_client.feed_batch(documents, self.TEST_SCHEMA)

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

        self._base_test_feed_batch_successful(self.client.feed_batch_sync, documents)

    def test_feed_batch_sync_emptyBatch_successful(self):
        documents = []

        self._base_test_feed_batch_successful(self.client.feed_batch_sync, documents)

    def test_feed_batch_multithreaded_successful(self):
        documents = [
            VespaDocument(id="doc1", fields={"title": "Title 1", "contents": "Content 1"}),
            VespaDocument(id="doc2", fields={"title": "Title 2"}),
        ]

        self._base_test_feed_batch_successful(self.client.feed_batch_multithreaded, documents)

    def test_feed_batch_multithreaded_emptyBatch_successful(self):
        documents = []

        self._base_test_feed_batch_successful(self.client.feed_batch_multithreaded, documents)

    def test_query_found_successful(self):
        documents = [
            {"id": "doc1", "fields": {"title": "Title 1", "contents": "Content 1"}},
            {"id": "doc2", "fields": {"title": "Title 1", "contents": "Content 1.1"}},
            {"id": "doc3", "fields": {"title": "Title 2"}}
        ]
        self.pyvespa_client.feed_batch(documents, self.TEST_SCHEMA)

        result = self.client.query(
            yql="select * from sources * where title contains 'Title 1';",
            ranking="bm25",
            model_restrict=self.TEST_SCHEMA
        )

        self.assertEqual(len(result.root.children), 2)

        titles = set([child.fields["title"] for child in result.root.children])
        contents = set([child.fields["contents"] for child in result.root.children])

        self.assertEqual(titles, {"Title 1"})
        self.assertEqual(contents, {"Content 1", "Content 1.1"})

    def test_query_notFound_successful(self):
        documents = [
            {"id": "doc1", "fields": {"title": "Title 1", "contents": "Content 1"}},
            {"id": "doc2", "fields": {"title": "Title 2"}}
        ]
        self.pyvespa_client.feed_batch(documents, self.TEST_SCHEMA)

        result = self.client.query(
            yql="select * from sources * where title contains 'Title 3';",
            ranking="bm25",
            model_restrict=self.TEST_SCHEMA
        )

        self.assertEqual(len(result.root.children), 0)

    def test_query_invalidQueryUrl_fails(self):
        query_client = VespaClient("http://localhost:8080", "http://localhost:8080", "http://localhost:8000")

        with self.assertRaises(VespaError):
            query_client.query(
                yql="select * from sources * where title contains 'Title 1';"
            )

    @patch.object(httpx, "get", wraps=httpx.get)
    @patch.object(httpx, "post", wraps=httpx.post)
    def test_download_application_successful(self, mock_post, mock_get):
        app = self.client.download_application()

        self.assertTrue(os.path.exists(app), "Application root does not exist")
        self.assertTrue(os.path.isfile(os.path.join(app, "services.xml")),
                        "services.xml does not exist or is not a file")
        self.assertTrue(os.path.isdir(os.path.join(app, "schemas")), "schemas does not exist or is not a directory")
        self.assertTrue(os.path.isfile(os.path.join(app, "schemas", "test_vespa_client.sd")),
                        "test_vespa_client.sd does not exist or is not a file")

    def test_download_application_createSessionError_fails(self):
        """
        Test that download_application fails when session creation fails
        """
        original_post = httpx.post

        def modified_post(*args, **kwargs):
            resp = original_post(*args, **kwargs)
            resp.status_code = 500
            return resp

        with patch.object(httpx, "post", new=modified_post):
            with self.assertRaises(VespaError):
                self.client.download_application()

    def test_download_application_downloadError_fails(self):
        original_get = httpx.get

        def modified_get(*args, **kwargs):
            resp = original_get(*args[1:], **kwargs)  # 1:0 to skip self argument
            resp.status_code = 500
            return resp

        with patch.object(httpx.Client, "get", new=modified_get):
            with self.assertRaises(VespaError):
                self.client.download_application()

    def test_deploy_application_successful(self):
        """
        Test that deploy_application works. To ensure we're not changing our local Vespa, we download the current
        application and deploy it. This means this test fails if donwload_application fails, even though we're not
        testing that here.
        """

        def get_vespa_app_generation() -> int:
            """
            Get the current Vespa application generation
            """
            resp = httpx.get("http://localhost:19071/application/v2/tenant/default/application/default")
            return resp.json()["generation"]

        app = self.client.download_application()

        with patch.object(httpx, "post", wraps=httpx.post) as mock_post:
            generation_before = get_vespa_app_generation()

            self.client.deploy_application(app)

            generation_after = get_vespa_app_generation()

            self.assertTrue(generation_after > generation_before)  # note generation can increase by more than 1
            mock_post.assert_called_once()
            self.assertTrue('prepareandactivate' in mock_post.call_args[0][0])

    def test_deploy_application_invalidAppPath_fails(self):
        with self.assertRaises(VespaError):
            self.client.deploy_application("/invalid/path")

    def test_deploy_application_invalidApp_fails(self):
        with self.assertRaises(VespaError):
            self.client.deploy_application(os.path.abspath(os.path.curdir))
