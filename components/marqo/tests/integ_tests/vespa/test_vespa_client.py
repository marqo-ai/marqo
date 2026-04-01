from importlib import reload

import asyncio
import functools
import httpcore
import httpx
import json
import orjson
import os
import pytest
import sys
import time
import unittest
import vespa.application as pyvespa
from unittest.mock import Mock, patch

from marqo.tensor_search.api import generate_config
from marqo.tensor_search.enums import EnvVars
from marqo.vespa import concurrency
from marqo.vespa.exceptions import (VespaError, VespaNotConvergedError,
                                    VespaStatusError, VespaTimeoutError)
from marqo.vespa.models import QueryResult, VespaDocument
from marqo.vespa.models.application_metrics import ApplicationMetrics
from marqo.vespa.models.query_result import Error
from marqo.vespa.vespa_client import VespaClient
from tests.integ_tests.marqo_test import AsyncMarqoTestCase


class TestVespaClient(AsyncMarqoTestCase):
    TEST_SCHEMA = "test_vespa_client"
    TEST_CLUSTER = "content_default"

    def setUp(self):
        self.client = VespaClient("http://localhost:19071", "http://localhost:8080",
                                  "http://localhost:8080", "content_default")
        self.pyvespa_client = pyvespa.Vespa(url="http://localhost", port=8080)
        self.test_document_1 = VespaDocument(id="doc1", fields={"title": "Title 1", "contents": "Content 1"})
        self.test_document_2 = VespaDocument(id="doc2", fields={"title": "Title 2", "contents": "Content 2"})
        self.pyvespa_client.delete_all_docs(self.TEST_CLUSTER, self.TEST_SCHEMA)

    def _base_test_feed_batch_successful(self, func, batch):
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
        path_ids = [response.path_id.split("/")[-1] for response in batch_response.responses]
        ids = [response.id.split("::")[-1] for response in batch_response.responses if response.status == 200]
        messages = [response.message for response in batch_response.responses]

        self.assertEqual(statuses, [200, 400])
        self.assertEqual(path_ids, ["doc1", "doc2"])
        self.assertEqual(ids, ["doc1"])
        self.assertIsNone(messages[0])
        self.assertIsNotNone(messages[1])

    def test_feed_batch_invalidFeedUrl_fails(self):
        feed_client = VespaClient("http://localhost:8080", "http://localhost:8009",
                                  "http://localhost:8080", "content_default")
        documents = [
            VespaDocument(id="doc1", fields={"title": "Title 1", "contents": "Content 1"}),
            VespaDocument(id="doc2", fields={"title": "Title 2"}),
        ]

        res = feed_client.feed_batch(documents, self.TEST_SCHEMA)
        self.assertEqual(2, len(res.responses))
        for r in res.responses:
            self.assertEqual(500, r.status)
            self.assertIn("Network Error", r.message)

    @pytest.mark.asyncio
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

    def test_delete_document_successful(self):
        documents = [
            {"id": "doc1", "fields": {"title": "Title 1", "contents": "Content 1"}},
            {"id": "doc2", "fields": {"title": "Title 2", "contents": "Content 2"}}
        ]
        self.pyvespa_client.feed_iterable(documents, self.TEST_SCHEMA)

        resp = self.client.delete_document("doc1", self.TEST_SCHEMA)

        self.assertEqual(resp.path_id.split("/")[-1], "doc1")
        self.assertEqual(resp.id.split("::")[-1], "doc1")

        # Verify document deleted

        get_responses = [
            self.pyvespa_client.get_data(data_id=data["id"], schema=self.TEST_SCHEMA) for data in
            [{"id": "doc1"}, {"id": "doc2"}]
        ]
        status = [{resp.json['id'].split('::')[-1]: resp.status_code} for resp in get_responses]

        self.assertEqual(status, [{"doc1": 404}, {"doc2": 200}])

    def test_delete_document_notFound_successful(self):
        documents = [
            {"id": "doc1", "fields": {"title": "Title 1", "contents": "Content 1"}},
        ]
        self.pyvespa_client.feed_iterable(documents, self.TEST_SCHEMA)

        # Note it's still 200 if the document doesn't exist
        resp = self.client.delete_document("docx", self.TEST_SCHEMA)

        self.assertEqual(resp.path_id.split("/")[-1], "docx")
        self.assertEqual(resp.id.split("::")[-1], "docx")

        # Verify document deleted
        get_responses = [
            self.pyvespa_client.get_data(data_id=data["id"], schema=self.TEST_SCHEMA) for data in  [{"id": "docx"}, {"id": "doc1"}]
        ]
        status = [{resp.json['id'].split('::')[-1]: resp.status_code} for resp in get_responses]

        self.assertEqual(status, [{"docx": 404}, {"doc1": 200}])

    def test_query_found_successful(self):
        documents = [
            {"id": "doc1", "fields": {"title": "Title 1", "contents": "Content 1"}},
            {"id": "doc2", "fields": {"title": "Title 1", "contents": "Content 1.1"}},
            {"id": "doc3", "fields": {"title": "Title 2"}}
        ]
        self.pyvespa_client.feed_iterable(documents, self.TEST_SCHEMA)

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
        self.pyvespa_client.feed_iterable(documents, self.TEST_SCHEMA)

        result = self.client.query(
            yql="select * from sources * where title contains 'Title 3';",
            ranking="bm25",
            model_restrict=self.TEST_SCHEMA
        )

        self.assertIsNone(result.root.children)

    def test_query_invalidQueryUrl_fails(self):
        query_client = VespaClient("http://localhost:8080", "http://localhost:8080",
                                   "http://localhost:8000", "content_default")

        with self.assertRaises(VespaError):
            query_client.query(
                yql="select * from sources * where title contains 'Title 1';"
            )

    def test_query_timeout_fails(self):
        """
        VespaTimeoutError error is raised when Vespa responds with a timeout error.
        """
        query_client = VespaClient("http://localhost:8080", "http://localhost:8080",
                                   "http://localhost:8080", "content_default")

        def modified_post(*args, **kwargs):
            kwargs['json']['timeout'] = '1ms'
            return httpx.post(*args, **kwargs)

        with patch.object(
                httpx.Client, "post",
                wraps=modified_post
        ):
            with self.assertRaisesStrict(VespaTimeoutError):
                query_client.query(
                    yql="select * from sources * where title contains 'Title 1';"
                )

    def test_default_search_timeout_fails(self):
        """
        VespaTimeoutError error is raised when VespaClient is created with a default timeout of 1ms.
        This will fail even if query 'timeout' isn't set, since the default timeout will be used.
        """
        query_client = VespaClient("http://localhost:8080", "http://localhost:8080",
                                   "http://localhost:8080", "content_default",
                                   default_search_timeout_ms=1)

        def pass_through_post(*args, **kwargs):
            return httpx.post(*args, **kwargs)

        with patch.object(
                httpx.Client, "post",
                wraps=pass_through_post
        ) as mock_post:
            with self.assertRaisesStrict(VespaTimeoutError):
                query_client.query(
                    yql="select * from sources * where title contains 'Title 1';"
                )
            # Ensure that post was called with correct timeout
            self.assertEqual(mock_post.call_args.kwargs['json']['timeout'], '1ms')

    def test_query_softDoom_fails(self):
        """
        VespaTimeoutError error is raised when Vespa responds with a soft doom error.
        """
        query_client = VespaClient("http://localhost:8080", "http://localhost:8080",
                                   "http://localhost:8080", "content_default")

        def modified_post(*args, **kwargs):
            resp = httpx.post(*args, **kwargs)
            result = QueryResult(**resp.json())
            result.root.errors = []
            result.root.errors.append(
                Error(
                    code=8,
                    summary='Error in search reply.',
                    message='Search request soft doomed during query setup and initialization.'
                )
            )
            return httpx.Response(
                status_code=504,
                content=result.json(by_alias=True).encode("utf-8"),
                request=resp.request
            )

        with patch.object(
                httpx.Client, "post",
                wraps=modified_post
        ):
            with self.assertRaisesStrict(VespaTimeoutError):
                query_client.query(
                    yql="select * from sources * where title contains 'Title 1';"
                )

    def test_query_nonHandled_fails(self):
        """
        VespaStatusError error is raised when Vespa responds with a non-handled error status code.
        """
        query_client = VespaClient("http://localhost:8080", "http://localhost:8080",
                                   "http://localhost:8080", "content_default")

        statuses = [400, 500, 504]

        def modified_post(*args, **kwargs):
            status = kwargs.get("status", 200)
            del kwargs["status"]
            resp = httpx.post(*args, **kwargs)
            resp.status_code = status
            return resp

        for status in statuses:
            with self.subTest(status):
                with patch.object(
                        httpx.Client, "post",
                        wraps=functools.partial(modified_post, status=status)
                ):
                    with self.assertRaisesStrict(VespaStatusError):
                        query_client.query(
                            yql="select * from sources * where title contains 'Title 1';"
                        )

    def test_download_application_successful(self):
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
        original_post = httpx.Client.post

        def modified_post(*args, **kwargs):
            resp = original_post(*args, **kwargs)
            resp.status_code = 500
            return resp

        with patch.object(httpx.Client, "post", new=modified_post):
            with self.assertRaises(VespaError):
                self.client.download_application()

    def test_download_application_downloadError_fails(self):
        original_get = httpx.Client.get

        def modified_get(*args, **kwargs):
            resp = original_get(*args, **kwargs)  # 1:0 to skip self argument
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

        with patch.object(httpx.Client, "post", wraps=httpx.post) as mock_post:
            generation_before = get_vespa_app_generation()

            self.client.deploy_application(app)

            generation_after = get_vespa_app_generation()

            self.assertTrue(generation_after > generation_before)  # note generation can increase by more than 1
            mock_post.assert_called_once()
            self.assertTrue('prepareandactivate' in mock_post.call_args[0][0])

    def test_deploy_application_invalidAppPath_fails(self):
        with self.assertRaises(VespaError):
            self.client.deploy_application("/invalid/path")

    @unittest.skip
    def test_deploy_application_invalidApp_fails(self):
        with self.assertRaises(VespaError):
            self.client.deploy_application(os.path.abspath(os.path.curdir))

    def test_feed_batch_documents_DocumentTimeOut_response_format(self):
        documents = [
            VespaDocument(id="doc1", fields={"title": "Title 1", "contents": "Content 1"}),
            VespaDocument(id="doc2", fields={"title": "Title 2"}),
        ]

        with patch("marqo.vespa.vespa_client.httpx.AsyncClient.post",
                   side_effect = httpx.TimeoutException("Timeout")):
            batch_response = self.client.feed_batch(documents, self.TEST_SCHEMA)

        self.assertEqual(batch_response.errors, True)
        self.assertEqual(2, len(batch_response.responses))
        for r in batch_response.responses:
            self.assertEqual(r.status, 500)
            self.assertIn("Network Error", r.message)

    def test_get_vespa_version(self):
        expected_vespa_version = '8.513.17'
        version = self.client.get_vespa_version()
        self.assertEqual(expected_vespa_version, version)

    def test_translate_vespa_document_response_status(self):
        test_cases = [
            (200, 200, None),
            (404, 404, "Document does not exist in the index"),
            (412, 400, "Marqo vector store couldn't update the document"),
            (429, 429, "Marqo vector store received too many requests. Please try again later"),
            (507, 400, "Marqo vector store is out of memory or disk space"),
            (123, 500, "Marqo vector store returned an unexpected error with this document"),
            (400, 500, "Marqo vector store returned an unexpected error with this document"),
            # generic 400 error without specific message
            (400, 400, "The document contains invalid characters in the fields. Original error: could not parse field"),
            # specific 400 error
        ]
        for status, expected_status, expected_message in test_cases:
            with self.subTest(status=status):
                if status == 400 and "could not parse field" in expected_message:
                    result_status, result_message = self.client.translate_vespa_document_response(
                        status, "could not parse field")
                else:
                    result_status, result_message = self.client.translate_vespa_document_response(
                        status,None)
                self.assertEqual(result_status, expected_status)
                if expected_message:
                    self.assertIn(expected_message, result_message)

    def test_translate_vespa_document_response_logging(self):
        with patch("marqo.vespa.vespa_client.logger.error") as mock_log_error:
            status = 400
            self.client.translate_vespa_document_response(status, None)
        mock_log_error.assert_called_once()

    @patch.object(
        VespaClient,
        '_get_convergence_status',
        side_effect=httpx._exceptions.ReadTimeout("Read Timeout")
    )
    def test_vespa_client_timeout_exception_handled(self, mock_get_convergence_status):
        """If a timeout exception is raised, the method should retry until the total wait time is reached"""
        side_effects = [
            httpx._exceptions.ReadTimeout("Read Timeout"),
            httpcore._exceptions.ReadTimeout("Read Timeout")
        ]
        for side_effect in side_effects:
            mock_get_convergence_status.side_effect = side_effect
            mock_get_convergence_status.reset_mock()
            with self.subTest(side_effect=side_effect):
                vespa_client = VespaClient("http://localhost:19071", "http://localhost:8080",
                                           "http://localhost:8080", "content_default")
                with self.assertRaises(VespaError) as e:
                    vespa_client.wait_for_application_convergence(0.1)
                self.assertIn("Vespa application did not converge", str(e.exception))

    @patch.object(VespaClient, '_get_convergence_status',
                  return_value=VespaClient._ConvergenceStatus(
                      current_generation=1, wanted_generation=2, converged=False))
    def test_application_convergence_timeout_fails(self, mock_get_convergence_status):
        """If the total wait time is reached, the method should raise a VespaError"""
        vespa_client = VespaClient("http://localhost:19071", "http://localhost:8080",
                                   "http://localhost:8080", "content_default")

        with self.assertRaises(VespaError) as e:
            vespa_client.wait_for_application_convergence(timeout=0.1)
        self.assertIn("Vespa application did not converge",str(e.exception))

    def test_get_pool_size_initialization(self):
        """Test that get_pool_size is properly set and used as default concurrency"""
        get_pool_size = 15
        client = VespaClient(
            "http://localhost:19071", 
            "http://localhost:8080",
            "http://localhost:8080", 
            "content_default",
            get_pool_size=get_pool_size
        )
        
        # Verify get_pool_size is set correctly
        self.assertEqual(client.get_pool_size, get_pool_size)
        
        client.close()

    @patch('marqo.vespa.vespa_client.httpx.AsyncClient')
    def test_get_batch_uses_correct_concurrency(self, mock_async_client_class):
        """Test that get_batch uses get_pool_size as default concurrency"""
        # Create a proper mock for the async client and response
        mock_async_client = mock_async_client_class.return_value.__aenter__.return_value
        
        # Mock the response object
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = orjson.dumps({
            'pathId': '/document/v1/test_vespa_client/test_vespa_client/docid/doc1',
            'id': 'test::doc1',
            'fields': {'title': 'Test'}
        })

        # Make the async client's get method return the mock response
        mock_async_client.get.return_value = mock_response

        # Feed a document first to ensure the schema exists
        test_doc = VespaDocument(id="doc1", fields={"title": "Test Title"})
        self.client.feed_document(test_doc, self.TEST_SCHEMA)

        # Call get_batch
        self.client.get_batch(['doc1'], self.TEST_SCHEMA)

        # Verify AsyncClient was created
        mock_async_client_class.assert_called_once()

    @patch('marqo.vespa.vespa_client.asyncio.Semaphore')
    @patch('marqo.vespa.vespa_client.httpx.AsyncClient')
    def test_get_batch_semaphore_uses_concurrency_parameter(self, mock_async_client_class, mock_semaphore_class):
        """Test that get_batch creates semaphore with concurrency parameter, using get_pool_size as default"""
        # Create a proper mock for the async client and response
        mock_async_client = mock_async_client_class.return_value.__aenter__.return_value
        
        # Mock the response object
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = orjson.dumps({
            'pathId': '/document/v1/test_vespa_client/test_vespa_client/docid/doc1',
            'id': 'test::doc1',
            'fields': {'title': 'Test'}
        })

        # Make the async client's get method return the mock response
        mock_async_client.get.return_value = mock_response

        # Create client with specific get_pool_size
        get_pool_size = 20
        client = VespaClient(
            "http://localhost:19071",
            "http://localhost:8080", 
            "http://localhost:8080",
            "content_default",
            get_pool_size=get_pool_size
        )
        
        # Test 1: Use default concurrency (should use get_pool_size)
        client.get_batch(['doc1'], self.TEST_SCHEMA)
        mock_semaphore_class.assert_called_with(get_pool_size)
        
        # Test 2: Use explicit concurrency parameter
        custom_concurrency = 8
        mock_semaphore_class.reset_mock()
        client.get_batch(['doc1'], self.TEST_SCHEMA, concurrency=custom_concurrency)
        mock_semaphore_class.assert_called_with(custom_concurrency)
        
        client.close()

    def test_vespa_client_close_calls_http_client_close(self):
        """Test that VespaClient.close() calls http_client.close()"""
        # Create a VespaClient with correct parameters
        client = VespaClient(
            config_url="http://localhost:19071",
            document_url="http://localhost:8080",
            query_url="http://localhost:8080",
            content_cluster_name="test_cluster",
            get_pool_size=10
        )
        
        # Mock the http_client.close method
        with patch.object(client.http_client, 'close') as mock_close:
            # Call close method
            client.close()
            
            # Verify that http_client.close was called
            mock_close.assert_called_once()

    def test_vespa_get_pool_size_env_var(self):
        """Test that VESPA_GET_POOL_SIZE environment variable properly sets get_pool_size"""
        
        # Test 1: Default value (should be 10)
        with self.subTest("default_get_pool_size"):
            config = generate_config()
            vespa_client = config.vespa_client
            
            # Verify default get_pool_size is used
            self.assertEqual(vespa_client.get_pool_size, 10)
            vespa_client.close()
        
        # Test 2: Set environment variable to custom value
        with self.subTest("custom_get_pool_size"):
            with patch.dict(os.environ, {EnvVars.VESPA_GET_POOL_SIZE: "25"}):
                # Import and call generate_config to create VespaClient with env var
                
                config = generate_config()
                vespa_client = config.vespa_client
                
                # Verify the custom get_pool_size is used
                self.assertEqual(vespa_client.get_pool_size, 25)
                
                vespa_client.close()

    def test_deploy_session(self):
        """Test that create_deployment_session calls the correct methods"""
        vespa_client = VespaClient("http://localhost:19071", "http://localhost:8080",
                                   "http://localhost:8080", "content_default")
        # wait for vespa to get converged
        vespa_client.wait_for_application_convergence(timeout=10)
        generation_before_deployment = vespa_client.get_application_generation()
        deployment_session = vespa_client.create_deployment_session()

        prep = vespa_client.prepare(deployment_session[1], 10)
        session_id = prep['session-id']
        self.assertEqual(prep['message'], f"Session {session_id} for tenant 'default' prepared.")

        activate = vespa_client.activate(prep['activate'], 10)
        previous_expected_generation = activate['application']['previousActiveGeneration']
        self.assertEqual(activate['message'], f"Session {session_id} for tenant 'default' activated.")

        self.assertEqual(previous_expected_generation, generation_before_deployment)
        base_expected_deploy_url = f"http://localhost:19071/application/v2/tenant/default/session/{session_id}/"
        self.assertEqual(deployment_session[0], base_expected_deploy_url + 'content/')
        self.assertEqual(deployment_session[1], base_expected_deploy_url + 'prepared')

        vespa_client.wait_for_application_convergence(timeout=10)
        generation_after_deployment = vespa_client.get_application_generation()
        self.assertEqual(generation_after_deployment, int(session_id))

    @patch.object(VespaClient, '_get_convergence_status',
                  return_value=VespaClient._ConvergenceStatus(
                      current_generation=1, wanted_generation=2, converged=False))
    def test_check_for_application_convergence_not_converged(self, mock_get_convergence_status):
        """Test that check_for_application_convergence raises an error if the application has not converged"""
        vespa_client = VespaClient("http://localhost:19071", "http://localhost:8080",
                                   "http://localhost:8080", "content_default")
        with self.assertRaises(VespaNotConvergedError) as e:
            vespa_client.check_for_application_convergence()
        self.assertIn("Vespa application has not converged", str(e.exception))

    def test_get_application_generation(self):
        """Test that get_application_generation returns the current application generation"""
        vespa_client = VespaClient("http://localhost:19071", "http://localhost:8080",
                                   "http://localhost:8080", "content_default")
        application_generation = vespa_client.get_application_generation()
        self.assertEqual(type(application_generation), int)

    def test_manage_content(self):
        deployment_session = self.client.create_deployment_session()
        content_base_url = deployment_session[0]

        self.client.put_content(content_base_url, "test", "test")

        list_content = self.client.list_contents(content_base_url)
        self.assertIn(f"{content_base_url}test", list_content)

        text_content = self.client.get_text_content(content_base_url, "test")
        self.assertEqual(text_content, "test")

        binary_content = self.client.get_binary_content(content_base_url, "test")
        self.assertEqual(binary_content, b"test")

        self.client.delete_content(content_base_url, "test")
        list_content = self.client.list_contents(content_base_url)
        self.assertNotIn(f"{content_base_url}test", list_content)

    def test_get_metrics(self):
        metrics = self.client.get_metrics()
        self.assertIsInstance(metrics, ApplicationMetrics)

    def test_feed_documents(self):
        self.client.feed_document(self.test_document_1, self.TEST_SCHEMA)
        get_response = self.client.get_document(
            id=self.test_document_1.id,
            schema=self.TEST_SCHEMA
        )
        document = get_response.document
        self.assertEqual(document.id, f'id:{self.TEST_SCHEMA}:{self.TEST_SCHEMA}::{self.test_document_1.id}')
        self.assertEqual(document.fields, self.test_document_1.fields)
        delete_response = self.client.delete_document(self.test_document_1.id, self.TEST_SCHEMA)
        self.assertEqual(
            delete_response.path_id,
            f'/document/v1/{self.TEST_SCHEMA}/{self.TEST_SCHEMA}/docid/{self.test_document_1.id}'
        )

        get_all_docs = self.client.get_all_documents(self.TEST_SCHEMA)
        self.assertEqual(get_all_docs.document_count, 0) # validate that all documents were deleted

    def test_delete_all_documents(self):
        response = self.client.get_all_documents(self.TEST_SCHEMA)
        self.assertEqual(response.document_count, 0)

        self.client.feed_document(self.test_document_1, self.TEST_SCHEMA)
        self.client.feed_document(self.test_document_2, self.TEST_SCHEMA)
        response = self.client.delete_all_docs(self.TEST_SCHEMA)
        self.assertEqual(response.document_count, 2)

        # Check it was all deleted
        response = self.client.get_all_documents(self.TEST_SCHEMA)
        self.assertEqual(response.document_count, 0)

    def test_get_all_documents(self):
        get_documents_response = self.client.get_all_documents(self.TEST_SCHEMA)
        self.assertEqual(get_documents_response.document_count, 0)

        # Feed 2 documents
        self.client.feed_document(self.test_document_1, self.TEST_SCHEMA)
        self.client.feed_document(self.test_document_2, self.TEST_SCHEMA)
        get_documents_response = self.client.get_all_documents(self.TEST_SCHEMA, stream=True)

        # Check retrieved documents match
        self.assertEqual(get_documents_response.document_count, 2)
        self.assertEqual(
            get_documents_response.documents[0].id,
            f'id:{self.TEST_SCHEMA}:{self.TEST_SCHEMA}::{self.test_document_1.id}'
        )
        self.assertEqual(
            get_documents_response.documents[1].id,
            f'id:{self.TEST_SCHEMA}:{self.TEST_SCHEMA}::{self.test_document_2.id}'
        )

    def test_batch_index_requests(self):
        feed_batch_docs = [
            VespaDocument(id="batch_doc1", fields={"title": "Title 1", "contents": "Content 1"}),
            VespaDocument(id="batch_doc2", fields={"title": "Title 2"}),
        ]

        batch_response = self.client.feed_batch(feed_batch_docs, self.TEST_SCHEMA)
        self.assertEqual(batch_response.errors, False)

        get_batch_response = self.client.get_batch(
            ids=["batch_doc1", "batch_doc2"],
            schema=self.TEST_SCHEMA
        )
        self.assertEqual(get_batch_response.errors, False)
        self.assertEqual(len(get_batch_response.responses), 2)

        get_batch_no_ids_response = self.client.get_batch(
            ids=[],
            schema=self.TEST_SCHEMA
        )
        self.assertEqual(get_batch_no_ids_response.errors, False)
        self.assertEqual(get_batch_no_ids_response.responses, [])

        delete_batch_response = self.client.delete_batch(
            ids=["batch_doc1", "batch_doc2"],
            schema=self.TEST_SCHEMA
        )
        self.assertEqual(delete_batch_response.errors, False)
        self.assertEqual(len(delete_batch_response.responses), 2)

        # Try getting documents after deletion
        get_batch_response = self.client.get_batch(
            ids=["batch_doc1", "batch_doc2"],
            schema=self.TEST_SCHEMA
        )
        self.assertEqual(get_batch_response.errors, True)
        self.assertEqual(len(get_batch_response.responses), 2)
        self.assertEqual(get_batch_response.responses[0].status, 404)
        self.assertEqual(get_batch_response.responses[1].status, 404)

    def test_get_batch_with_fields_parameter(self):
        """Test that get_batch with fields parameter only returns requested fields"""
        feed_batch_docs = [
            VespaDocument(id="fields_doc1", fields={"title": "Title 1", "contents": "Content 1"}),
            VespaDocument(id="fields_doc2", fields={"title": "Title 2", "contents": "Content 2"}),
        ]

        batch_response = self.client.feed_batch(feed_batch_docs, self.TEST_SCHEMA)
        self.assertEqual(batch_response.errors, False)

        # Get batch with only 'title' field
        get_batch_response = self.client.get_batch(
            ids=["fields_doc1", "fields_doc2"],
            schema=self.TEST_SCHEMA,
            fields=["title"]
        )
        self.assertEqual(get_batch_response.errors, False)
        self.assertEqual(len(get_batch_response.responses), 2)
        for response in get_batch_response.responses:
            self.assertEqual(response.status, 200)
            self.assertIn("title", response.document.fields)
            self.assertNotIn("contents", response.document.fields)

    def test_get_document_async_with_specific_fields_deserializes_response(self):
        """Test that _get_document_async_with_specific_fields correctly deserializes the response using orjson"""
        feed_docs = [VespaDocument(id="specific_fields_doc1", fields={"title": "Title 1", "contents": "Content 1"})]
        self.client.feed_batch(feed_docs, self.TEST_SCHEMA)

        async def _run():
            async with httpx.AsyncClient() as async_client:
                semaphore = asyncio.Semaphore(1)
                return await self.client._get_document_async_with_specific_fields(
                    semaphore, async_client, "specific_fields_doc1", ["title"], self.TEST_SCHEMA, 60
                )

        result = asyncio.run(_run())
        self.assertEqual(result.status, 200)
        self.assertIn("title", result.document.fields)
        self.assertNotIn("contents", result.document.fields)

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.put", return_value=httpx.Response(status_code=200, content="Invalid JSON"))
    async def test_update_document_json_decode_error(self, mock_put):
        """
        Test that _update_document_async properly raises VespaError on JSONDecodeError for a 200 response.
        """

        document = VespaDocument(
            id="doc1", fields={
                "title": "Updated Title"
            }
            )

        with pytest.raises(VespaError, match="Unexpected response from Vespa"):
            await self.client._update_document_async(
                asyncio.Semaphore(1), httpx.AsyncClient(), document, "test_schema", 60, "marqo__id"
                )

        assert mock_put.call_count == 1

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.post", return_value=httpx.Response(status_code=200, content="Invalid JSON"))
    async def test_feed_document_json_decode_error(self, mock_post):
        """
        Test that feed_document properly raises VespaError on JSONDecodeError for a 200 response.
        """

        document = VespaDocument(
            id="doc1", fields={
                "title": "Updated Title"
            }
            )

        with pytest.raises(VespaError, match="Unexpected response from Vespa"):
            await self.client._feed_document_async(
                asyncio.Semaphore(1), httpx.AsyncClient(), document, "test_schema", 60
                )

        assert mock_post.call_count == 1

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient.delete", return_value=httpx.Response(status_code=200, content="Invalid JSON"))
    async def test_delete_document_json_decode_error(self, mock_delete):
        """
        Test that delete_document properly raises VespaError on JSONDecodeError for a 200 response.
        """

        with pytest.raises(VespaError, match="Unexpected response: Invalid JSON"):
            await self.client._delete_document_async(
                asyncio.Semaphore(1), httpx.AsyncClient(), "doc1", "test_schema", 60
                )

        assert mock_delete.call_count == 1

    def test_httpx_client_should_not_timeout_before_vespa_timeout(self):
        """Test that httpx client will not time out before Vespa timeout"""
        def delayed_vespa_504(*args, **kwargs):
            time.sleep(7)  # emulate Vespa taking ~7s
            payload = {
                "root": {
                    "relevance": 0.0,
                    "errors": [{
                        "code": 8,
                        "summary": "Error in search reply.",
                        "message": "Search request soft doomed during query setup and initialization."
                    }]
                }
            }
            req = httpx.Request("POST", "http://dummy/search/")
            return httpx.Response(status_code=504, content=json.dumps(payload).encode(), request=req)

        query_client = VespaClient("http://localhost:8080","http://localhost:8080",
                                   "http://localhost:8080","content_default")

        with patch.object(httpx.Client, "post", side_effect=delayed_vespa_504):
            with self.assertRaisesStrict(VespaTimeoutError):
                query_client.query(
                    yql="select * from sources * where title contains 'Title 1';",
                    timeout=7000 # 7 seconds Vespa timeout, 8 seconds httpx timeout
                )

    def test_marqo_search_random_connection_close_rate_1(self):
        """Test that query sends 'Connection: close' header based on the configured random close rate."""
        documents = [
            {"id": "doc1", "fields": {"title": "Title 1", "contents": "Content 1"}},
        ]
        self.pyvespa_client.feed_iterable(documents, self.TEST_SCHEMA)

        with self.help_mock_environment_variables_in_settings({"MARQO_SEARCH_RANDOM_CONNECTION_CLOSE_RATE": "1.0"}):
            reload(sys.modules["marqo.vespa.vespa_client"])  # reload module to apply env var change
            with patch.object(httpx.Client, "post", wraps=httpx.post) as mock_post:
                self.client.query(
                    yql="select * from sources * where title contains 'Title 1';",
                    model_restrict=self.TEST_SCHEMA
                )
                headers = mock_post.call_args.kwargs.get("headers") or {}
                self.assertEqual("close", headers.get("Connection"), )

    def test_marqo_search_random_connection_close_rate_0(self):
        with self.help_mock_environment_variables_in_settings({"MARQO_SEARCH_RANDOM_CONNECTION_CLOSE_RATE": "0.0"}):
            reload(sys.modules["marqo.vespa.vespa_client"])
            for _ in range(10):  # run multiple times to check that Connection: close is not sent
                with patch.object(httpx.Client, "post", wraps=httpx.post) as mock_post:
                    self.client.query(
                        yql="select * from sources * where title contains 'Title 1';",
                        model_restrict=self.TEST_SCHEMA
                    )
                    headers = mock_post.call_args.kwargs.get("headers")
                    if headers:
                        self.assertNotIn("Connection", headers)

    def test_marqo_search_random_connection_close_rate_0_1(self):
        with self.help_mock_environment_variables_in_settings({"MARQO_SEARCH_RANDOM_CONNECTION_CLOSE_RATE": "0.3"}):
            reload(sys.modules["marqo.vespa.vespa_client"])
            counter = 0
            for seed in range(10):  # run multiple times to check that Connection: close is not sent
                with patch.object(httpx.Client, "post", wraps=httpx.post) as mock_post:
                    self.client.query(
                        yql="select * from sources * where title contains 'Title 1';",
                        model_restrict=self.TEST_SCHEMA, drop_connection_random_seed=seed
                    )
                    headers = mock_post.call_args.kwargs.get("headers")
                    if headers and headers.get("Connection") == "close":
                        counter += 1
            self.assertEqual(4, counter)

