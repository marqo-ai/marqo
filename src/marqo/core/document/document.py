from timeit import default_timer as timer
from typing import Dict, List, Tuple, Optional

import marqo.api.exceptions as api_exceptions
from marqo.core.constants import MARQO_DOC_ID
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.exceptions import UnsupportedFeatureError, ParsingError, InternalError
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models.marqo_add_documents_response import MarqoAddDocumentsResponse, MarqoAddDocumentsItem
from marqo.core.models.marqo_index import IndexType, SemiStructuredMarqoIndex, StructuredMarqoIndex, \
    UnstructuredMarqoIndex
from marqo.core.models.marqo_update_documents_response import MarqoUpdateDocumentsResponse, MarqoUpdateDocumentsItem
from marqo.core.semi_structured_vespa_index.semi_structured_add_document_handler import \
    SemiStructuredAddDocumentsHandler, SemiStructuredFieldCountConfig
from marqo.core.structured_vespa_index.structured_add_document_handler import StructuredAddDocumentsHandler
from marqo.core.unstructured_vespa_index.unstructured_add_document_handler import UnstructuredAddDocumentsHandler
from marqo.core.vespa_index.vespa_index import for_marqo_index as vespa_index_factory
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

    def add_documents(self, add_docs_params: AddDocsParams,
                      field_count_config=SemiStructuredFieldCountConfig()) -> MarqoAddDocumentsResponse:
        marqo_index = self.index_management.get_index(add_docs_params.index_name)

        if isinstance(marqo_index, StructuredMarqoIndex):
            add_docs_handler = StructuredAddDocumentsHandler(marqo_index, add_docs_params, self.vespa_client)
        elif isinstance(marqo_index, SemiStructuredMarqoIndex):
            add_docs_handler = SemiStructuredAddDocumentsHandler(marqo_index, add_docs_params,
                                                                 self.vespa_client, self.index_management,
                                                                 field_count_config)
        elif isinstance(marqo_index, UnstructuredMarqoIndex):
            add_docs_handler = UnstructuredAddDocumentsHandler(marqo_index, add_docs_params, self.vespa_client)
        else:
            raise InternalError(f"Unknown index type {type(marqo_index)}")

        return add_docs_handler.add_documents()

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
        if marqo_index.type in [IndexType.Unstructured, IndexType.SemiStructured]:
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
                status, message = self.vespa_client.translate_vespa_document_response(resp.status, None)
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
                status, message = self.vespa_client.translate_vespa_document_response(resp.status, resp.message)
                new_item = MarqoAddDocumentsItem(id=doc_id, status=status, message=message)
                new_items.append(new_item)

        for loc, error_info in unsuccessful_docs:
            new_items.insert(loc, error_info)
            errors = True

        return MarqoAddDocumentsResponse(errors=errors, index_name=index_name, items=new_items,
                                         processingTimeMs=add_docs_processing_time_ms)
