from timeit import default_timer as timer
from typing import Dict, List, Tuple, Optional

import marqo.api.exceptions as api_exceptions
from marqo.core.constants import MARQO_DOC_ID
from marqo.core.exceptions import UnsupportedFeatureError, ParsingError
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models.marqo_add_documents_response import MarqoAddDocumentsResponse, MarqoAddDocumentsItem
from marqo.core.models.marqo_index import IndexType
from marqo.core.models.marqo_update_documents_response import MarqoUpdateDocumentsResponse, MarqoUpdateDocumentsItem
from marqo.core.vespa_index import for_marqo_index as vespa_index_factory
from marqo.logging import get_logger
from marqo.vespa.models import UpdateDocumentsBatchResponse, VespaDocument
from marqo.vespa.models.delete_document_response import DeleteAllDocumentsResponse
from marqo.vespa.models.feed_response import FeedBatchResponse
from marqo.vespa.vespa_client import VespaClient

logger = get_logger(__name__)


class Document:
    """A class that handles the document API in Marqo"""

    def __init__(self, vespa_client: VespaClient, index_management: IndexManagement):
        self.vespa_client = vespa_client
        self.index_management = index_management

    def delete_all_docs_by_index_name(self, index_name: str) -> int:
        """Delete all documents in the given index by index name.

        Args:
            index_name: The name of the index to delete documents from"""
        marqo_index = self.index_management.get_index(index_name)
        return self.delete_all_docs(marqo_index)

    def delete_all_docs(self, marqo_index) -> int:
        """Delete all documents in the given index by marqo_index object.

        Args:
            marqo_index: The index object to delete documents from"""
        res: DeleteAllDocumentsResponse = self.vespa_client.delete_all_docs(marqo_index.schema_name)
        return res.document_count

    def partial_update_documents_by_index_name(self, index_name,
                                               partial_documents: List[Dict]) \
            -> MarqoUpdateDocumentsResponse:
        """Partially update documents in the given index by index name.

        Args:
            index_name: The name of the index to partially update documents in
            partial_documents: A list of documents to partially update

        Raises:
            IndexNotFoundError: If the index does not exist

        Return:
            A MarqoUpdateDocumentsResponse containing the response of the partial update operation
        """
        marqo_index = self.index_management.get_index(index_name)

        return self.partial_update_documents(partial_documents, marqo_index)

    def partial_update_documents(self, partial_documents: List[Dict], marqo_index) \
            -> MarqoUpdateDocumentsResponse:
        """Partially update documents in the given index by marqo_index object.

        The partial_documents without _id will error out and the error will be returned in the response without
        error out the entire batch.

        If there exists duplicate _id in the partial_documents, the last document will be used.

        If the document does not exist, this document will error out and the error will be returned in the response.

        Args:
            partial_documents: A list of documents to partially update
            marqo_index: The index object to partially update documents in

        Raises:
            UnsupportedFeatureError: If the index is unstructured

        Return:
            MarqoUpdateDocumentsResponse containing the response of the partial update operation
        """
        if marqo_index.type == IndexType.Unstructured:
            raise UnsupportedFeatureError("Partial document update is not supported for unstructured indexes. "
                                          "Please use add_documents with use_existing_tensor=True instead")
        elif marqo_index.type == IndexType.Structured:
            pass
        else:
            raise ValueError(f"Invalid index type: {marqo_index.type}")

        start_time = timer()
        vespa_index = vespa_index_factory(marqo_index)
        vespa_documents: List[VespaDocument] = []
        unsuccessful_docs: List[Tuple[int, MarqoUpdateDocumentsItem]] = []

        # Remove duplicated documents based on _id
        partial_documents, _ = self.remove_duplicated_documents(partial_documents)

        for index, doc in enumerate(partial_documents):
            try:
                vespa_document = VespaDocument(**vespa_index.to_vespa_partial_document(doc))
                vespa_documents.append(vespa_document)
            except ParsingError as e:
                unsuccessful_docs.append(
                    (index, MarqoUpdateDocumentsItem(id=doc.get(MARQO_DOC_ID, ''), error=e.message,
                                                     status=int(api_exceptions.InvalidArgError.status_code))))

        vespa_res: UpdateDocumentsBatchResponse = (
            self.vespa_client.update_documents_batch(vespa_documents,
                                                     marqo_index.schema_name,
                                                     vespa_id_field=vespa_index.get_vespa_id_field()))

        return self._translate_update_document_response(vespa_res, unsuccessful_docs,
                                                        marqo_index.name, start_time)

    def _translate_update_document_response(self, responses: UpdateDocumentsBatchResponse, unsuccessful_docs: List,
                                            index_name: str, start_time) \
            -> MarqoUpdateDocumentsResponse:
        """Translate Vespa response dict into MarqoUpdateDocumentsResponse for document update.

        Args:
            responses: The response from Vespa
            unsuccessful_docs: The list of unsuccessful documents
            index_name: The name of the index
            start_time: The start time of the operation

        Return:
            MarqoUpdateDocumentsResponse containing the response of the partial update operation
        """

        items: List[MarqoUpdateDocumentsItem] = []

        errors = responses.errors

        if responses is not None:
            for resp in responses.responses:
                doc_id = resp.id.split('::')[-1] if resp.id else None
                status, message = self.translate_vespa_document_response(resp.status)
                new_item = MarqoUpdateDocumentsItem(id=doc_id, status=status, message=message, error=message)
                items.append(new_item)

        for loc, error_info in unsuccessful_docs:
            items.insert(loc, error_info)
            errors = True

        return MarqoUpdateDocumentsResponse(errors=errors, index_name=index_name, items=items,
                                            processingTimeMs=(timer() - start_time) * 1000)

    def remove_duplicated_documents(self, documents: List) -> Tuple[List, set]:
        """Remove duplicated documents based on _id in the given list of documents.

        For a list of documents, if there exists duplicate _id, the last document will be used while the
        previous ones will be removed from the list.

        This function does not validate the documents, it only removes the duplicates based on _id fields.
        """
        # Deduplicate docs, keep the latest
        docs = []
        doc_ids = set()
        for i in range(len(documents) - 1, -1, -1):
            doc = documents[i]

            if isinstance(doc, dict) and '_id' in doc:
                doc_id = doc['_id']
                try:
                    if doc_id is not None and doc_id in doc_ids:
                        logger.debug(f'Duplicate document ID {doc_id} found, keeping the latest')
                        continue
                    doc_ids.add(doc_id)
                except TypeError as e:  # Happens if ID is a non-hashable type -- ID validation will catch this later on
                    logger.debug(f'Could not hash document ID {doc_id}: {e}')

            docs.append(doc)
        # Reverse to preserve order in request
        docs.reverse()
        return docs, doc_ids

    def translate_add_documents_response(self, responses: Optional[FeedBatchResponse],
                                         index_name: str,
                                         unsuccessful_docs: List,
                                         add_docs_processing_time_ms: float) \
            -> MarqoAddDocumentsResponse:
        """Translate Vespa FeedBatchResponse into MarqoAddDocumentsResponse.

        Args:
            responses: The response from Vespa
            index_name: The name of the index
            unsuccessful_docs: The list of unsuccessful documents
            add_docs_processing_time_ms: The processing time of the add documents operation, in milliseconds

        Return:
            MarqoAddDocumentsResponse: The response of the add documents operation
        """

        new_items: List[MarqoAddDocumentsItem] = []
        # A None response means no documents are sent to Vespa. Probably all documents are invalid and blocked in Marqo.
        errors = responses.errors if responses is not None else True

        if responses is not None:
            for resp in responses.responses:
                doc_id = resp.id.split('::')[-1] if resp.id else None
                status, message = self.translate_vespa_document_response(resp.status, message=resp.message)
                new_item = MarqoAddDocumentsItem(id=doc_id, status=status, message=message)
                new_items.append(new_item)

        for loc, error_info in unsuccessful_docs:
            new_items.insert(loc, error_info)
            errors = True

        return MarqoAddDocumentsResponse(errors=errors, index_name=index_name, items=new_items,
                                         processingTimeMs=add_docs_processing_time_ms)

    def translate_vespa_document_response(self, status: int, message: Optional[str]=None) -> Tuple[int, Optional[str]]:
        """A helper function to translate Vespa document response into the expected status, message that
        is used in Marqo document API responses.

        Args:
            status: The status code from Vespa document response

        Return:
            A tuple of status code and the message in the response
        """
        if status == 200:
            return 200, None
        elif status == 404:
            return 404, "Document does not exist in the index"
        # Update documents get 412 from Vespa for document not found as we use condition
        elif status == 412:
            return 404, "Document does not exist in the index"
        elif status == 429:
            return 429, "Marqo vector store receives too many requests. Please try again later"
        elif status == 507:
            return 400, "Marqo vector store is out of memory or disk space"
        # TODO Block the invalid special characters before sending to Vespa
        elif status == 400 and isinstance(message, str) and "could not parse field" in message.lower():
            return 400, f"The document contains invalid characters in the fields. Original error: {message} "
        else:
            logger.error(f"An unexpected error occurred from the Vespa document response. "
                         f"status: {status}, message: {message}")
            return 500, f"Marqo vector store returns an unexpected error with this document. Original error: {message}"