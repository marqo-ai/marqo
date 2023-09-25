import asyncio
import concurrent
from typing import Dict, Any, List

import httpx

import marqo.vespa.concurrency as conc
from marqo.vespa.models import *
from marqo.vespa.models.feed_response import BatchFeedResponse


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
        self.query_url = query_url
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

        resp = self.http_client.post(f'{self.query_url}/search/', data=query)

        resp.raise_for_status()

        return QueryResult(**resp.json())

    def feed_batch(self,
                   batch: List[VespaDocument],
                   schema: str,
                   concurrency: int = 100,
                   timeout: int = 60) -> List[FeedResponse]:
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
        responses = conc.run_coroutine(
            self._feed_batch_async(batch, schema, concurrency, timeout)
        )

        return responses

    async def _feed_batch_async(self, batch: List[VespaDocument],
                                schema: str,
                                connections: int, timeout: int) -> BatchFeedResponse:
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

        return BatchFeedResponse(responses=responses, errors=errors)

    async def _feed_document_async(self, semaphore: asyncio.Semaphore, async_client: httpx.AsyncClient,
                                   document: VespaDocument, schema: str,
                                   timeout: int) -> FeedResponse:
        doc_id = document.id
        data = {'fields': document.fields}

        async with semaphore:
            end_point = f"{self.document_url}/document/v1/{schema}/{schema}/docid/{doc_id}"
            resp = await async_client.post(end_point, json=data, timeout=timeout)

        try:
            # This will cover 200 and document-specific errors. Other unexpected errors will be raised.
            return FeedResponse(**resp.json(), status=resp.status_code)
        except:
            resp.raise_for_status()

    def feed_batch_sync(self, batch: List[Dict[str, Any]], schema: str) -> List[FeedResponse]:
        """
        Feed a batch of documents to Vespa sequentially.

        Args:
            batch: List of documents to feed
            schema: Schema to feed to

        Returns:
            List of FeedResponse objects
        """
        pass


if __name__ == '__main__':
    client = VespaClient('', 'http://a90106143149745d1a731f29fa145882-2096005137.us-east-1.elb.amazonaws.com:8080',
                         'http://a90106143149745d1a731f29fa145882-2096005137.us-east-1.elb.amazonaws.com:8080')

    # r = client.query(
    #     yql='select * from marqo_settings where true',
    #     hits=100
    # )
    r = client.feed_batch([
        VespaDocument(id='test1', fields={}),
        VespaDocument(id='test2', fields={'random': 'test'}),
    ], schema='marqo_settings')

    pass
