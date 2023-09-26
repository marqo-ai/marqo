import asyncio
import random
import time
from concurrent.futures import ThreadPoolExecutor
from json import JSONDecodeError
from typing import Dict, Any, List

import httpx

import marqo.logging
import marqo.vespa.concurrency as conc
from marqo.vespa.exceptions import VespaStatusError, VespaError
from marqo.vespa.models import *
from marqo.vespa.models.feed_response import FeedBatchResponse

logger = marqo.logging.get_logger(__name__)


class VespaClient:
    def __init__(self, config_url: str, document_url: str, query_url: str, pool_size: int = 10):
        """
        Create a VespaClient object.
        Args:
            config_url: Vespa Deploy API base URL
            document_url: Vespa Document API base URL
            query_url: Vespa Query API base URL
            pool_size: Number of connections to keep in the connection pool
        """
        self.config_url = config_url.strip('/')
        self.document_url = document_url.strip('/')
        self.query_url = query_url.strip('/')
        self.http_client = httpx.Client(
            limits=httpx.Limits(max_keepalive_connections=pool_size, max_connections=pool_size)
        )

    def close(self):
        """
        Close the VespaClient object.
        """
        self.http_client.close()

    def deploy_application(self, application_zip: str):
        """
        Deploy a Vespa application.
        Args:
            application_zip: Path to the Vespa application zip file
        """
        pass

    def download_application(self):
        pass

    def query(self, yql: str, hits: int = 10, ranking: str = None, model_restrict: str = None,
              query_features: Dict[str, Any] = None, **kwargs) -> QueryResult:
        """
        Query Vespa.
        Args:
            yql: YQL query
            hits: Number of hits to return
            ranking: Ranking profile to use
            model_restrict: Schema to restrict the query to
            query_features: Query features
            **kwargs: Additional query parameters
        Returns:
            Query result as a VespaQueryResult object
        """
        query_features_list = {
            f'input.query({key})': value for key, value in query_features.items()
        } if query_features else {}
        query = {
            'yql': yql,
            'hits': hits,
            'ranking': ranking,
            'model.restrict': model_restrict,
            **query_features_list,
            **kwargs
        }
        query = {key: value for key, value in query.items() if value}

        logger.debug(f'Query: {query}')

        resp = self.http_client.post(f'{self.query_url}/search/', data=query)

        resp.raise_for_status()

        return QueryResult(**resp.json())

    def feed_batch(self,
                   batch: List[VespaDocument],
                   schema: str,
                   concurrency: int = 100,
                   timeout: int = 60) -> FeedBatchResponse:
        """
        Feed a batch of documents to Vespa concurrently.

        Documents will be fed in batches of `batch_size` documents, with `concurrency` concurrent pooled connections.

        Args:
            batch: List of documents to feed
            schema: Schema to feed to
            concurrency: Number of concurrent feed requests
            timeout: Timeout in seconds per request

        Returns:
            List of FeedResponse objects
        """
        if not batch:
            return FeedBatchResponse(responses=[], errors=False)

        batch_response = conc.run_coroutine(
            self._feed_batch_async(batch, schema, concurrency, timeout)
        )

        return batch_response

    def feed_batch_sync(self, batch: List[VespaDocument], schema: str) -> FeedBatchResponse:
        """
        Feed a batch of documents to Vespa sequentially.

        This method is for debugging and experimental purposes only. Sequential feeding can be very slow.

        Args:
            batch: List of documents to feed
            schema: Schema to feed to

        Returns:
            List of FeedResponse objects
        """
        with httpx.Client(limits=httpx.Limits(max_keepalive_connections=10, max_connections=10)) as sync_client:
            responses = [
                self._feed_document_sync(sync_client, document, schema, timeout=60)
                for document in batch
            ]

        errors = False
        for response in responses:
            if response.status != "200":
                errors = True

        return FeedBatchResponse(responses=responses, errors=errors)

    def feed_batch_multithreaded(self, batch: List[VespaDocument], schema: str,
                                 max_threads: int = 100) -> FeedBatchResponse:
        """
        Feed a batch of documents to Vespa concurrently using a thread pool.

        This method is for debugging and experimental purposes only. Use `feed_batch` instead to feed documents
        asynchronously with one thread.

        Args:
            batch: List of documents to feed
            schema: Schema to feed to
            max_threads: Maximum number of threads to use

        Returns:
            List of FeedResponse objects
        """
        with httpx.Client(
                limits=httpx.Limits(max_keepalive_connections=max_threads, max_connections=max_threads)) as sync_client:
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                responses = list(executor.map(
                    lambda document: self._feed_document_sync(sync_client, document, schema, timeout=60), batch
                ))

        errors = False
        for response in responses:
            if response.status != "200":
                errors = True

        return FeedBatchResponse(responses=responses, errors=errors)

    async def _feed_batch_async(self, batch: List[VespaDocument],
                                schema: str,
                                connections: int, timeout: int) -> FeedBatchResponse:
        async with httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=connections,
                                                         max_connections=connections)) as async_client:
            semaphore = asyncio.Semaphore(connections)
            tasks = [
                asyncio.create_task(
                    self._feed_document_async(semaphore, async_client, document, schema, timeout)
                )
                for document in batch
            ]
            await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

        responses = []
        errors = False
        for task in tasks:
            result = task.result()
            responses.append(task.result())
            if result.status != "200":
                errors = True

        return FeedBatchResponse(responses=responses, errors=errors)

    async def _feed_document_async(self, semaphore: asyncio.Semaphore, async_client: httpx.AsyncClient,
                                   document: VespaDocument, schema: str,
                                   timeout: int) -> FeedResponse:
        doc_id = document.id
        data = {'fields': document.fields}

        async with semaphore:
            end_point = f"{self.document_url}/document/v1/{schema}/{schema}/docid/{doc_id}"
            try:
                resp = await async_client.post(end_point, json=data, timeout=timeout)
            except httpx.HTTPError as e:
                raise VespaError(e) from e

        try:
            # This will cover 200 and document-specific errors. Other unexpected errors will be raised.
            return FeedResponse(**resp.json(), status=resp.status_code)
        except JSONDecodeError:
            self._raise_for_status(resp)

    def _feed_document_sync(self, sync_client: httpx.Client, document: VespaDocument, schema: str,
                            timeout: int) -> FeedResponse:
        doc_id = document.id
        data = {'fields': document.fields}

        end_point = f"{self.document_url}/document/v1/{schema}/{schema}/docid/{doc_id}"

        resp = sync_client.post(end_point, json=data, timeout=timeout)

        return FeedResponse(**resp.json(), status=resp.status_code)

    def _raise_for_status(self, resp):
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise VespaStatusError(e) from e


if __name__ == '__main__':
    client = VespaClient('', 'http://a90106143149745d1a731f29fa145882-2096005137.us-east-1.elb.amazonaws.com:8080',
                         'http://a90106143149745d1a731f29fa145882-2096005137.us-east-1.elb.amazonaws.com:8080')

    count = 100
    concurrency = 10

    random_vectors = [
        [random.uniform(0, 1) for _ in range(384)] for _ in range(count)
    ]
    batch = [
        VespaDocument(
            id=f'test{i}',
            fields={
                'title': f'Test title{i}',
                'marqo_embeddings_title': {
                    "0": random_vectors[i]
                }
            })
        # VespaDocument(id='test2', fields={'random': 'test'}),
        for i in range(count)
    ]
    #
    # start = time.time()
    # r = client.feed_batch_multithreaded(
    #     batch,
    #     schema='simplewiki',
    #     max_threads=concurrency
    # )
    # end = time.time()
    #
    # print(f'Multithreaded time: {end - start} seconds')
    # print(f'Errors: {r.errors}')

    start = time.time()
    r = client.feed_batch(
        batch,
        schema='simplewiki',
        concurrency=concurrency
    )
    end = time.time()

    print(f'Async time: {end - start} seconds')
    print(f'Errors: {r.errors}')
