from timeit import default_timer as timer
from typing import Dict, List, Tuple, Optional, Collection, Union

import marqo.api.exceptions as api_exceptions
from marqo.core.constants import MARQO_DOC_ID, MARQO_DOC_TENSORS
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.exceptions import UnsupportedFeatureError, ParsingError, InternalError
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models.marqo_add_documents_response import MarqoAddDocumentsResponse, MarqoAddDocumentsItem
from marqo.core.models.marqo_get_documents_by_id_response import MarqoGetDocumentsByIdsResponse, \
    MarqoGetDocumentsByIdsItem
from marqo.core.models.marqo_index import IndexType, SemiStructuredMarqoIndex, StructuredMarqoIndex, \
    UnstructuredMarqoIndex, MarqoIndex
from marqo.core.models.marqo_update_documents_response import MarqoUpdateDocumentsResponse, MarqoUpdateDocumentsItem
from marqo.core.semi_structured_vespa_index.semi_structured_add_document_handler import \
    SemiStructuredAddDocumentsHandler, SemiStructuredFieldCountConfig
from marqo.core.structured_vespa_index.structured_add_document_handler import StructuredAddDocumentsHandler
from marqo.core.unstructured_vespa_index.common import MARQO_DOC_MULTIMODAL_PARAMS
from marqo.core.unstructured_vespa_index.unstructured_add_document_handler import UnstructuredAddDocumentsHandler
from marqo.core.vespa_index.vespa_index import for_marqo_index as vespa_index_factory
from marqo.logging import get_logger
from marqo.tensor_search import validation
from marqo.tensor_search.enums import TensorField
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

    def get_documents_by_ids(self, marqo_index: MarqoIndex, document_ids: Collection[str],
                             show_vectors: bool = False, ignore_invalid_ids: bool = False
    ) -> MarqoGetDocumentsByIdsResponse:
        """
        Returns documents by their IDs.

        Args:
            ignore_invalid_ids: If True, invalid IDs will be ignored and not returned in the response. If False, an error
                will be raised if any of the IDs are invalid
        """
        if not isinstance(document_ids, Collection):
            raise api_exceptions.InvalidArgError("Get documents must be passed a collection of IDs!")
        if len(document_ids) <= 0:
            raise api_exceptions.InvalidArgError("Can't get empty collection of IDs!")

        # max_docs_limit = utils.read_env_vars_and_defaults(EnvVars.MARQO_MAX_RETRIEVABLE_DOCS)
        # if max_docs_limit is not None and len(document_ids) > int(max_docs_limit):
        #     raise api_exceptions.IllegalRequestedDocCount(
        #         f"{len(document_ids)} documents were requested, which is more than the allowed limit of [{max_docs_limit}], "
        #         f"set by the environment variable `{EnvVars.MARQO_MAX_RETRIEVABLE_DOCS}`")

        unsuccessful_docs: List[Tuple[int, MarqoGetDocumentsByIdsItem]] = []

        validated_ids = []
        for loc, doc_id in enumerate(document_ids):
            try:
                validated_ids.append(validation.validate_id(doc_id))
            except api_exceptions.InvalidDocumentIdError as e:
                if not ignore_invalid_ids:
                    unsuccessful_docs.append(
                        (
                            loc, MarqoGetDocumentsByIdsItem(
                                # Invalid IDs are not returned in the response
                                id=doc_id,
                                message=e.message,
                                status=int(e.status_code)
                            )
                        )
                    )
                else:
                    logger.debug(f'Invalid document ID {doc_id} ignored')

        if len(validated_ids) == 0:  # Can only happen when ignore_invalid_ids is True
            return MarqoGetDocumentsByIdsResponse(errors=True, results=[i[1] for i in unsuccessful_docs])

        batch_get = self.vespa_client.get_batch(validated_ids, marqo_index.schema_name)
        vespa_index = vespa_index_factory(marqo_index)

        results: List[Union[MarqoGetDocumentsByIdsItem, Dict]] = []
        errors = batch_get.errors

        for response in batch_get.responses:
            if response.status == 200:
                marqo_document = vespa_index.to_marqo_document(response.document.dict())
                # if show_vectors:
                #     if constants.MARQO_DOC_TENSORS in marqo_document:
                #         marqo_document[TensorField.tensor_facets] = _get_tensor_facets(
                #             marqo_document[constants.MARQO_DOC_TENSORS])
                #     else:
                #         marqo_document[TensorField.tensor_facets] = []

                if not show_vectors:
                    if MARQO_DOC_MULTIMODAL_PARAMS in marqo_document:
                        del marqo_document[MARQO_DOC_MULTIMODAL_PARAMS]

                if MARQO_DOC_TENSORS in marqo_document:
                    del marqo_document[MARQO_DOC_TENSORS]

                results.append(
                    {
                        TensorField.found: True,
                        **marqo_document
                    }
                )
            else:
                status, message = self.vespa_client.translate_vespa_document_response(response.status, None)
                results.append(
                    MarqoGetDocumentsByIdsItem(
                        id=self._get_id_from_vespa_id(response.id), status=status,
                        found=False, message=message)
                )

        # Insert the error documents at the correct locations
        for loc, error_info in unsuccessful_docs:
            results.insert(loc, error_info)
            errors = True

        return MarqoGetDocumentsByIdsResponse(errors=errors, results=results)

    def _get_id_from_vespa_id(self, vespa_id: str) -> str:
        """Returns the document ID from a Vespa ID. Vespa IDs are of the form `namespace::document_id`."""
        return vespa_id.split('::')[-1]

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
        if marqo_index.type in [IndexType.Unstructured]:
            raise UnsupportedFeatureError("Partial document update is not supported for unstructured indexes. "
                                          "Please use add_documents with use_existing_tensor=True instead")
        elif marqo_index.type in [IndexType.Structured, IndexType.SemiStructured]:
            pass
        else:
            raise ValueError(f"Invalid index type: {marqo_index.type}")

        start_time = timer()
        vespa_index = vespa_index_factory(marqo_index)
        vespa_documents: List[VespaDocument] = []
        unsuccessful_docs: List[Tuple[int, MarqoUpdateDocumentsItem]] = []

        # Remove duplicated documents based on _id
        partial_documents, doc_ids = self.remove_duplicated_documents(partial_documents)

        existing_vespa_documents = []
        if marqo_index.type == IndexType.SemiStructured:
            get_batch_response = self.vespa_client.get_batch(list(doc_ids), marqo_index.schema_name)
            existing_vespa_documents = [doc_response.document for doc_response in get_batch_response.responses
                                        if doc_response.status == 200]

        existing_documents_map = {doc.id: doc.fields for doc in existing_vespa_documents}

        for index, doc in enumerate(partial_documents):
            try:
                vespa_document = VespaDocument(**vespa_index.to_vespa_partial_document(
                    doc, existing_documents_map.get(doc['_id'], None)))
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
