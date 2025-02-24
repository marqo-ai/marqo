"""
This file contains the `VespaIndexMixin` class, responsible for **all document-related operations**,
such as **retrieving, updating, feeding, and deleting documents** in a Vespa index.

Functions Included:
--------------------
1. **Feeding Documents**
   - `feed_document()` → Feed a single document to Vespa.
   - `feed_batch()` → Feed multiple documents asynchronously.
   - `feed_batch_sync()` → Feed documents sequentially (for debugging).
   - `feed_batch_multithreaded()` → Feed documents using multiple threads.

2. **Retrieving Documents**
   - `get_document()` → Fetch a document by ID.
   - `get_all_documents()` → Fetch all documents in a schema.
   - `get_batch()` → Fetch multiple documents concurrently.

3. **Deleting Documents**
   - `delete_document()` → Delete a document by ID.
   - `delete_all_docs()` → Delete all documents in a schema.
   - `delete_batch()` → Delete multiple documents concurrently.

4. **Updating Documents**
   - `update_documents_batch()` → Perform a **partial update** on a batch of documents.

This mixin **requires** the `http_client` and `document_url` attributes from `VespaClientBase` to send HTTP requests.

Usage Example:
--------------
vespa = VespaClient(...)
vespa.feed_document(document=my_doc, schema="products")
vespa.get_document(id="123", schema="products")
vespa.delete_document(id="123", schema="products")
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from json import JSONDecodeError
from typing import List, Optional, TYPE_CHECKING

import httpx

import marqo.logging
import marqo.vespa.concurrency as conc
from marqo.vespa.exceptions import (VespaStatusError, VespaError)
from marqo.vespa.models import VespaDocument, FeedBatchResponse, FeedDocumentResponse, UpdateDocumentsBatchResponse, \
    UpdateDocumentResponse, FeedBatchDocumentResponse
from marqo.vespa.models.delete_document_response import DeleteDocumentResponse, DeleteBatchDocumentResponse, \
    DeleteBatchResponse, DeleteAllDocumentsResponse
from marqo.vespa.models.get_document_response import GetDocumentResponse, VisitDocumentsResponse, GetBatchResponse, \
    GetBatchDocumentResponse

if TYPE_CHECKING:
    from ._client_base import VespaClientBase

logger = marqo.logging.get_logger(__name__)


class VespaIndexMixin:
    def feed_document(self: "VespaClientBase", document: VespaDocument, schema: str, timeout: int = 60) -> FeedDocumentResponse:
        """
        Feed a document to Vespa.

        Args:
            document: Document to feed
            schema: Schema to feed to
            timeout: Timeout in seconds

        Returns:
            FeedResponse object
        """
        doc_id = document.id
        data = {
            'fields': document.fields
        }

        end_point = f'{self.document_url}/document/v1/{schema}/{schema}/docid/{doc_id}'

        resp = self.http_client.post(end_point, json=data, timeout=timeout)

        self._raise_for_status(resp)

        return FeedDocumentResponse(**resp.json())

    def feed_batch(
            self, batch: List[VespaDocument], schema: str, concurrency: Optional[int] = None,
            timeout: int = 60) -> FeedBatchResponse:
        """
        Feed a batch of documents to Vespa concurrently.

        Documents will be fed with `concurrency` concurrent pooled connections.

        Args:
            batch: List of documents to feed
            schema: Schema to feed to
            concurrency: Number of concurrent feed requests
            timeout: Timeout in seconds per request

        Returns:
            A FeedBatchResponse object
        """
        if not batch:
            return FeedBatchResponse(responses=[], errors=False)

        if concurrency is None:
            concurrency = self.feed_pool_size

        batch_response = conc.run_coroutine(
            self._feed_batch_async(batch, schema, concurrency, timeout)
        )

        return batch_response

    def feed_batch_sync(self: "VespaClientBase", batch: List[VespaDocument], schema: str) -> FeedBatchResponse:
        """
        Feed a batch of documents to Vespa sequentially.

        This method is for debugging and experimental purposes only. Sequential feeding can be very slow.

        Args:
            batch: List of documents to feed
            schema: Schema to feed to

        Returns:
            List of FeedResponse objects
        """
        responses = [self._feed_document_sync(self.http_client, document, schema, timeout=60) for document in batch]

        errors = False
        for response in responses:
            if response.status != 200:
                errors = True

        return FeedBatchResponse(responses=responses, errors=errors)

    def feed_batch_multithreaded(
            self, batch: List[VespaDocument], schema: str, max_threads: int = 10) -> FeedBatchResponse:
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
                limits=httpx.Limits(max_keepalive_connections=max_threads, max_connections=max_threads)
        ) as sync_client:
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                responses = list(
                    executor.map(
                        lambda document: self._feed_document_sync(sync_client, document, schema, timeout=60), batch
                    )
                )

        errors = False
        for response in responses:
            if response.status != 200:
                errors = True

        return FeedBatchResponse(responses=responses, errors=errors)

    def get_document(self: "VespaClientBase", id: str, schema: str) -> GetDocumentResponse:
        """
        Get a document by ID.

        Args:
            id: Document ID
            schema: Schema to get from

        Returns:
            GetDocumentResponse object
        """
        try:
            resp = self.http_client.get(f'{self.document_url}/document/v1/{schema}/{schema}/docid/{id}')
        except httpx.HTTPError as e:
            raise VespaError(e) from e

        self._raise_for_status(resp)

        return GetDocumentResponse(**resp.json())

    def get_all_documents(
            self, schema: str, stream=False, continuation: Optional[str] = None) -> VisitDocumentsResponse:
        """
        Get all documents in a schema.
        Args:
            schema: Schema to get from
            stream: Whether to stream the response
            continuation: Continuation token for pagination

        Returns:
            BatchGetDocumentResponse object
        """
        try:
            url = self._add_query_params(
                url=f'{self.document_url}/document/v1/{schema}/{schema}/docid', query_params={
                    'stream': str(stream).lower(),
                    'continuation': continuation
                }
            )
            logger.debug(f'URL: {url}')
            resp = self.http_client.get(url)
        except httpx.HTTPError as e:
            raise VespaError(e) from e

        self._raise_for_status(resp)

        return VisitDocumentsResponse(**resp.json())

    def get_batch(
            self, ids: List[str], schema: str, concurrency: Optional[int] = None,
            timeout: int = 60) -> GetBatchResponse:
        """
        Get a batch of documents by ID concurrently.

        Documents will be fetched with `concurrency` concurrent pooled connections.

        Missing (404) documents will be returned in the response. Any other non-200 responses will raise an exception.

        Args:
            ids: List of document IDs to get
            schema: Schema to get from
            concurrency: Number of concurrent get requests
            timeout: Timeout in seconds per request

        Returns:
            List of GetDocumentResponse objects containing the documents fetched and any missing documents (404)
        """
        if not ids:
            return GetBatchResponse(responses=[], errors=False)

        if concurrency is None:
            concurrency = self.get_pool_size

        batch_response = conc.run_coroutine(
            self._get_batch_async(ids, schema, concurrency, timeout)
        )

        return batch_response

    def delete_document(self: "VespaClientBase", id: str, schema: str) -> DeleteDocumentResponse:
        """
        Delete a document by ID.

        Note that this method returns a successful response even if the document does not exist.

        Args:
            id: Document ID
            schema: Schema to delete from
        """
        try:
            resp = self.http_client.delete(f'{self.document_url}/document/v1/{schema}/{schema}/docid/{id}')
        except httpx.HTTPError as e:
            raise VespaError(e) from e

        self._raise_for_status(resp)

        return DeleteDocumentResponse(**resp.json())

    def delete_all_docs(self: "VespaClientBase", schema: str) -> DeleteAllDocumentsResponse:
        """Deletes all documents in the given index"""
        try:
            resp = self.http_client.delete(
                f'{self.document_url}/document/v1/{schema}'
                f'/{schema}/docid/?cluster={self.content_cluster_name}&selection=true'
                )
        except httpx.HTTPError as e:
            raise VespaError(e) from e

        self._raise_for_status(resp)
        return DeleteAllDocumentsResponse(**resp.json())

    def delete_batch(
            self, ids: List[str], schema: str, concurrency: Optional[int] = None,
            timeout: int = 60) -> DeleteBatchResponse:
        """
        Delete a batch of documents by ID concurrently.

        Documents will be deleted with `concurrency` concurrent pooled connections.

        Args:
            ids: List of document IDs to delete
            schema: Schema to delete from
            concurrency: Number of concurrent delete requests
            timeout: Timeout in seconds per request

        Returns:
            A DeleteBatchResponse object
        """
        if not ids:
            return DeleteBatchResponse(responses=[], errors=False)

        if concurrency is None:
            concurrency = self.delete_pool_size

        batch_response = conc.run_coroutine(
            self._delete_batch_async(ids, schema, concurrency, timeout)
        )

        return batch_response

    def update_documents_batch(
            self, batch: List[VespaDocument], schema: str, concurrency: Optional[int] = None, timeout: int = 60,
            vespa_id_field: str = "marqo__id") -> UpdateDocumentsBatchResponse:
        """
        Partial update documents in batch concurrently.

        If the document does not exist, it will not be created and an error for that document will be
        returned in the response.

        Args:
            batch: A list of documents to update
            schema: schema name
            concurrency: Number of concurrent delete requests, can be configured by environment variable
                VESPA_PARTIAL_UPDATE_POOL_SIZE
            timeout: Timeout in seconds per request
            vespa_id_field: The field name of the vespa document id under the fields dictionary

        Returns:
            A UpdateDocumentsBatchResponse object
        """

        if not batch:
            return UpdateDocumentsBatchResponse(responses=[], errors=False)

        if concurrency is None:
            concurrency = self.partial_pool_size

        batch_response = conc.run_coroutine(
            self._update_documents_batch_async(batch, schema, concurrency, timeout, vespa_id_field)
        )

        return batch_response

    async def _feed_batch_async(self: "VespaClientBase", batch: List[VespaDocument],
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
            responses.append(result)
            if result.status != 200:
                errors = True

        return FeedBatchResponse(responses=responses, errors=errors)

    async def _update_documents_batch_async(self: "VespaClientBase", batch: List[VespaDocument],
                                            schema: str,
                                            connections: int, timeout: int,
                                            vespa_id_field: str) -> UpdateDocumentsBatchResponse:
        async with httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=connections,
                                                         max_connections=connections)) as async_client:
            semaphore = asyncio.Semaphore(connections)
            tasks = [
                asyncio.create_task(
                    self._update_document_async(semaphore, async_client, document, schema, timeout, vespa_id_field)
                )
                for document in batch
            ]
            await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

        responses = []
        errors = False
        for task in tasks:
            result = task.result()
            responses.append(result)
            if result.status != 200:
                errors = True

        return UpdateDocumentsBatchResponse(responses=responses, errors=errors)

    async def _update_document_async(self: "VespaClientBase", semaphore: asyncio.Semaphore, async_client: httpx.AsyncClient,
                                     document: VespaDocument, schema: str,
                                     timeout: int, vespa_id_field: str) -> UpdateDocumentResponse:
        doc_id = document.id
        data = {'fields': document.fields}

        # only used for documents that are not updated
        error_doc_path_id = f"/document/v1/{schema}/{schema}/docid/{doc_id}"

        async with semaphore:
            end_point = f'{self.document_url}/document/v1/{schema}/{schema}/docid/{doc_id}?create=false'
            data["condition"] = f'{schema}.{vespa_id_field}==\"{doc_id}\"'
            try:
                resp = await async_client.put(end_point, json=data, timeout=timeout)
            except httpx.RequestError as e:
                logger.error(e, exc_info=True)
                return UpdateDocumentResponse(status=500, message="Network Error", id=doc_id, path_id=error_doc_path_id)

        # Handle other exceptions
        try:
            return UpdateDocumentResponse(**resp.json(), status=resp.status_code)
        except JSONDecodeError as e:
            if resp.status_code == 200:
                # A 200 response shouldn't reach here, so we error out the whole batch
                raise VespaError(cause=e, message=f"Unexpected response from Vespa: {resp.text}") from e

            try:
                self._raise_for_status(resp)
            except VespaStatusError as e:
                logger.error(e, exc_info=True)
                return UpdateDocumentResponse(status=resp.status_code, message=e.message, id=doc_id,
                                              error_doc_path_id=error_doc_path_id)

    async def _feed_document_async(self: "VespaClientBase", semaphore: asyncio.Semaphore, async_client: httpx.AsyncClient,
                                   document: VespaDocument, schema: str,
                                   timeout: int) -> FeedBatchDocumentResponse:
        """An async method to feed a document to Vespa.

        Note: This method is used by the async feed batch method to feed documents concurrently. Unhandled exceptions
        will be raised in the main thread and leads a 500 error for the whole batch. Therefore, exceptions should be
        handled gracefully in this method for the specific document. We should keep the error message as similar as the
        Vespa error messages since this is a low level method. Overwrite the error message in higher level methods.

        Exceptions that are handled in this method:
        1. httpx.RequestError: We convert this error to a 500 error for the specific document and put 'Network Error' in
        the message.
        2. JSONDecodeError: If the Vespa response is 200 but the response can not be decoded, we raise a VespaError and
        this will block the whole batch as this indicates an unexpected response from Vespa.
        3. httpx.status_codes.HTTPStatusError: We catch the error and return it to marqo.core.document methods to handle
        it.

        Raises:
            VespaError: If the Vespa response is 200 but the response can not be decoded.

        Returns:
            FeedDocumentResponse object
        """
        doc_id = document.id
        data = {'fields': document.fields}

        async with semaphore:
            end_point = f'{self.document_url}/document/v1/{schema}/{schema}/docid/{doc_id}'
            # Handle httpx.RequestError
            try:
                resp = await async_client.post(end_point, json=data, timeout=timeout)
            except httpx.RequestError as e:
                logger.error(e, exc_info=True)
                return FeedBatchDocumentResponse(status=500, message="Network Error", id=doc_id)

        # Handle other exceptions
        try:
            return FeedBatchDocumentResponse(**resp.json(), status=resp.status_code)
        except JSONDecodeError as e:
            if resp.status_code == 200:
                # A 200 response shouldn't reach here, so we error out the whole batch
                raise VespaError(cause=e, message=f"Unexpected response from Vespa: {resp.text}") from e

            try:
                self._raise_for_status(resp)
            except VespaStatusError as e:
                logger.error(e, exc_info=True)
                return FeedBatchDocumentResponse(status=resp.status_code, message=e.message, id=doc_id)

    def _feed_document_sync(self: "VespaClientBase", sync_client: httpx.Client, document: VespaDocument, schema: str,
                            timeout: int) -> FeedBatchDocumentResponse:
        doc_id = document.id
        data = {'fields': document.fields}

        end_point = f'{self.document_url}/document/v1/{schema}/{schema}/docid/{doc_id}'

        resp = sync_client.post(end_point, json=data, timeout=timeout)

        return FeedBatchDocumentResponse(**resp.json(), status=resp.status_code)

    async def _get_batch_async(self: "VespaClientBase",
                               ids: List[str],
                               schema: str,
                               connections: int, timeout: int) -> GetBatchResponse:
        async with httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=connections,
                                                         max_connections=connections)) as async_client:
            semaphore = asyncio.Semaphore(connections)
            tasks = [
                asyncio.create_task(
                    self._get_document_async(semaphore, async_client, id, schema, timeout)
                )
                for id in ids
            ]
            await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

        responses = []
        errors = False
        for task in tasks:
            result = task.result()
            responses.append(result)
            if result.status != 200:
                errors = True

        return GetBatchResponse(responses=responses, errors=errors)

    async def _get_document_async(self: "VespaClientBase",
                                  semaphore: asyncio.Semaphore,
                                  async_client: httpx.AsyncClient,
                                  id: str,
                                  schema: str,
                                  timeout: int) -> GetBatchDocumentResponse:
        async with semaphore:
            try:
                resp = await async_client.get(
                    f'{self.document_url}/document/v1/{schema}/{schema}/docid/{id}', timeout=timeout
                )
            except httpx.HTTPError as e:
                raise VespaError(e) from e

            if resp.status_code in [200, 404]:
                return GetBatchDocumentResponse(**resp.json(), status=resp.status_code)

            self._raise_for_status(resp)

    async def _delete_batch_async(self: "VespaClientBase",
                                  ids: List[str],
                                  schema: str,
                                  connections: int, timeout: int) -> DeleteBatchResponse:
        async with httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=connections,
                                                         max_connections=connections)) as async_client:
            semaphore = asyncio.Semaphore(connections)
            tasks = [
                asyncio.create_task(
                    self._delete_document_async(semaphore, async_client, id, schema, timeout)
                )
                for id in ids
            ]
            await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

        responses = []
        errors = False
        for task in tasks:
            result = task.result()
            responses.append(result)
            if result.status != 200:
                errors = True

        return DeleteBatchResponse(responses=responses, errors=errors)

    async def _delete_document_async(self: "VespaClientBase",
                                     semaphore: asyncio.Semaphore,
                                     async_client: httpx.AsyncClient,
                                     id: str,
                                     schema: str,
                                     timeout: int) -> DeleteBatchDocumentResponse:
        async with semaphore:
            try:
                resp = await async_client.delete(f'{self.document_url}/document/v1/{schema}/{schema}/docid/{id}')
            except httpx.HTTPError as e:
                raise VespaError(e) from e

        try:
            # This will cover 200 and document-specific errors. Other unexpected errors will be raised.
            return DeleteBatchDocumentResponse(**resp.json(), status=resp.status_code)
        except JSONDecodeError:
            if resp.status_code == 200:
                # A 200 response shouldn't reach here
                raise VespaError(f'Unexpected response: {resp.text}')

            self._raise_for_status(resp)
