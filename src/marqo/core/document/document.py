from timeit import default_timer as timer
from typing import Dict, List, Tuple, Optional

import marqo.api.exceptions as api_exceptions
from marqo.core.constants import MARQO_DOC_ID
from marqo.core.exceptions import UnsupportedFeatureError, ParsingError, InternalError, MarqoDocumentParsingError
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.inference.api import Inference
from marqo.core.models.add_docs_params import AddDocsParams
from marqo.core.models.marqo_add_documents_response import MarqoAddDocumentsResponse, MarqoAddDocumentsItem
from marqo.core.models.marqo_index import IndexType, SemiStructuredMarqoIndex, StructuredMarqoIndex, \
    UnstructuredMarqoIndex
from marqo.core.models.marqo_update_documents_response import MarqoUpdateDocumentsResponse, MarqoUpdateDocumentsItem
from marqo.core.semi_structured_vespa_index.common import VESPA_FIELD_ID, INT_FIELDS, FLOAT_FIELDS, \
    VESPA_DOC_FIELD_TYPES, VESPA_DOC_VERSION_UUID
from marqo.core.semi_structured_vespa_index.semi_structured_add_document_handler import \
    SemiStructuredAddDocumentsHandler, SemiStructuredFieldCountConfig
from marqo.core.structured_vespa_index.structured_add_document_handler import StructuredAddDocumentsHandler
from marqo.core.unstructured_vespa_index.unstructured_add_document_handler import UnstructuredAddDocumentsHandler
from marqo.core.vespa_index.vespa_index import for_marqo_index as vespa_index_factory
from marqo.logging import get_logger
from marqo.marqo_docs import update_documents_response
from marqo.tensor_search.telemetry import RequestMetricsStore
from marqo.vespa.models import UpdateDocumentsBatchResponse, VespaDocument
from marqo.vespa.models.delete_document_response import DeleteAllDocumentsResponse
from marqo.vespa.models.feed_response import FeedBatchResponse
from marqo.vespa.vespa_client import VespaClient

logger = get_logger(__name__)


class Document:
    """A class that handles the document API in Marqo"""

    def __init__(self, vespa_client: VespaClient, index_management: IndexManagement, inference: Inference):
        self.vespa_client = vespa_client
        self.index_management = index_management
        self.inference = inference

    def add_documents(self, add_docs_params: AddDocsParams,
                      field_count_config=SemiStructuredFieldCountConfig()) -> MarqoAddDocumentsResponse:
        marqo_index = self.index_management.get_index(add_docs_params.index_name)

        if isinstance(marqo_index, StructuredMarqoIndex):
            add_docs_handler = StructuredAddDocumentsHandler(marqo_index, add_docs_params, self.vespa_client,
                                                             self.inference)
        elif isinstance(marqo_index, SemiStructuredMarqoIndex):
            add_docs_handler = SemiStructuredAddDocumentsHandler(marqo_index, add_docs_params,
                                                                 self.vespa_client, self.index_management,
                                                                 self.inference, field_count_config)
        elif isinstance(marqo_index, UnstructuredMarqoIndex):
            add_docs_handler = UnstructuredAddDocumentsHandler(marqo_index, add_docs_params, self.vespa_client,
                                                               self.inference)
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
            partial_documents: A list of documents to partially update received in the request
            marqo_index: The index object to partially update documents in

        Raises:
            UnsupportedFeatureError: If the index is unstructured

        Return:
            MarqoUpdateDocumentsResponse containing the response of the partial update operation
        """
        if marqo_index.type is IndexType.Unstructured:
            raise UnsupportedFeatureError("Partial document update is not supported for unstructured indexes. "
                                          "Please use add_documents with use_existing_tensor=True instead")
        elif marqo_index.type is IndexType.Structured:
            pass
        elif marqo_index.type is IndexType.SemiStructured:
            if not marqo_index.index_supports_partial_updates: # Partial updates for semi-structured indexes are only supported for Marqo version >= 2.16.0
                raise UnsupportedFeatureError(
                    f"The 'update_documents' endpoint for an unstructured index "
                    f"is only supported for indexes created with Marqo version 2.16.0 or "
                    f"later. This index was created with Marqo {marqo_index.version}."
                )
        else:
            raise ValueError(f"Invalid index type: {marqo_index.type}")

        start_time = timer()
        vespa_index = vespa_index_factory(marqo_index)
        vespa_documents: List[VespaDocument] = []
        unsuccessful_docs: List[Tuple[int, MarqoUpdateDocumentsItem]] = []

        # Remove duplicated documents based on _id
        partial_documents, documents_that_contain_maps = self.process_documents(partial_documents,
                                                                                         unsuccessful_docs, is_index_semi_structured=marqo_index.type is IndexType.SemiStructured)
        documents_that_contain_maps_but_dont_exist_in_vespa = set() # Set to keep track of documents that contain maps but don't exist in Vespa. This will remain unpopulated for structured indexes.
        existing_vespa_documents = {}

        # This block will only execute for SemiStructured Indexes and in case the request has documents that contain maps.
        # 1. Retrieve those documents, which contain maps in the update request, from Vespa.
        # 2. If there's any documents that dont' exist in Vespa, we will append them to unsuccessful_docs
        if marqo_index.type is IndexType.SemiStructured and documents_that_contain_maps: # Only retrieve the document back if the partial update request contains maps and the index is semi-structured
            get_batch_response = self.vespa_client.get_batch(ids = list(documents_that_contain_maps.keys()), fields = [
                VESPA_FIELD_ID, INT_FIELDS, FLOAT_FIELDS, VESPA_DOC_FIELD_TYPES, VESPA_DOC_VERSION_UUID], schema = marqo_index.schema_name)
            responses = get_batch_response.responses
            for resp in responses:
                if resp.document:
                    existing_vespa_documents[resp.document.fields[VESPA_FIELD_ID]] = resp.document.dict()
                else: 
                    id = self.extract_document_id_from_vespa_id(resp) # Extract the document id from the Vespa response. Vespa response will contain the document id even though the document was not found.
                    unsuccessful_docs.append((documents_that_contain_maps.get(id), MarqoUpdateDocumentsItem(id = id,
                                                                                                         status = int(api_exceptions.BadRequestError.status_code),
                                                                                                         error = "Marqo vector store couldn't update the document. Please see: " + update_documents_response() + " for more details")))
                    documents_that_contain_maps_but_dont_exist_in_vespa.add(id)

        for index, doc in enumerate(partial_documents):
            if (marqo_index.type is IndexType.SemiStructured # Only have this check in place for SemiStructured indexes
                    and documents_that_contain_maps_but_dont_exist_in_vespa
                    and doc.get(MARQO_DOC_ID, '') in documents_that_contain_maps_but_dont_exist_in_vespa):
                continue
            try:
                vespa_document = VespaDocument(**vespa_index.to_vespa_partial_document(doc, existing_vespa_documents.get(doc.get(MARQO_DOC_ID, ''), None)))
                vespa_documents.append(vespa_document)
            except ParsingError as e:
                unsuccessful_docs.append(
                    (index, MarqoUpdateDocumentsItem(id=doc.get(MARQO_DOC_ID, ''), error=e.message,
                                                     status=int(api_exceptions.InvalidArgError.status_code))))

        with RequestMetricsStore.for_request().time("partial_update.vespa._bulk"):
            vespa_res: UpdateDocumentsBatchResponse = (
                self.vespa_client.update_documents_batch(vespa_documents,
                                                         marqo_index.schema_name,
                                                         vespa_id_field=vespa_index.get_vespa_id_field()))

        with RequestMetricsStore.for_request().time("partial_update.postprocess"):
            result = self._translate_update_document_response(vespa_res, unsuccessful_docs,
                                                              marqo_index.name, start_time)

        return result

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
                doc_id = self.extract_document_id_from_vespa_id(resp)
                status, message = self.vespa_client.translate_vespa_document_response(resp.status, None)
                new_item = MarqoUpdateDocumentsItem(id=doc_id, status=status, message=message, error=message)
                items.append(new_item)

        for loc, error_info in unsuccessful_docs:
            items.insert(loc, error_info)
            errors = True

        return MarqoUpdateDocumentsResponse(errors=errors, index_name=index_name, items=items,
                                            processingTimeMs=(timer() - start_time) * 1000)

    def process_documents(self, documents: List[Dict], unsuccessful_docs: List[Tuple[int, MarqoUpdateDocumentsItem]],
                          is_index_semi_structured = False) -> Tuple[List[Dict], Dict]:
        """Process documents to remove duplicates and identify documents containing maps.
        
        This method combines duplicate removal and map detection into a single pass through
        the documents for better efficiency.

        Args:
            is_index_semi_structured: Variable denoting if the index that's is currently being processed is of type SemiStructured
            unsuccessful_docs: A list of documents which were processed unsuccessfully
            documents: List of document dictionaries to process
            
        Returns:
            Tuple containing:
            - List of deduplicated documents
            - Set of unique document IDs
            - Dictionary of document IDs that contain dictionary values and their index in the documents list
        """
        docs = []
        doc_ids = set()
        documents_with_maps = {}
        
        # Process documents in reverse to keep latest version of duplicates
        for i in range(len(documents) - 1, -1, -1):
            doc = documents[i]
            
            if not isinstance(doc, dict) or '_id' not in doc:
                docs.append(doc)
                continue
                
            doc_id = doc['_id']
            
            try:
                # Skip if we've already seen this ID
                if doc_id is not None and doc_id in doc_ids:
                    logger.debug(f'Duplicate document ID {doc_id} found, keeping the latest')
                    continue
                
                # Check for dictionary values while processing doc to populate the documents_with_maps set.
                # Only do it in case of semi-structured indexes.
                if is_index_semi_structured:
                    for field_name, field_value in doc.items():
                        if isinstance(field_value, dict):
                            if len(field_value) == 0: # If the dictionary is empty, get back the document so that we can update the doc with an empty dictionary (i.e remove the map from the doc).
                                documents_with_maps[doc_id] = i
                            else:
                                for key, val in field_value.items():
                                    if isinstance(val, (int, float)):
                                        documents_with_maps[doc_id] = i
                                        break
                                    else:
                                        raise MarqoDocumentParsingError(
                                            f'Unsupported field type {type(val)} for field {field_name} in doc {doc_id}. We only support int and float types for map values when updating a document.'
                                        )
                            break
                doc_ids.add(doc_id)
                docs.append(doc)
                
            except TypeError as e:
                logger.debug(f'Could not hash document ID {doc_id}: {e}')
                docs.append(doc)

            except MarqoDocumentParsingError as e:
                unsuccessful_docs.append((i, MarqoUpdateDocumentsItem(id=doc.get(MARQO_DOC_ID, ''),
                                                                      error=e.message,
                                                                      status=int(api_exceptions.InvalidArgError.status_code))))

        # Reverse to preserve original order
        docs.reverse()
        return docs, documents_with_maps

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
                doc_id = self.extract_document_id_from_vespa_id(resp)
                status, message = self.vespa_client.translate_vespa_document_response(resp.status, resp.message)
                new_item = MarqoAddDocumentsItem(id=doc_id, status=status, message=message)
                new_items.append(new_item)

        for loc, error_info in unsuccessful_docs:
            new_items.insert(loc, error_info)
            errors = True

        return MarqoAddDocumentsResponse(errors=errors, index_name=index_name, items=new_items,
                                         processingTimeMs=add_docs_processing_time_ms)

    def extract_document_id_from_vespa_id(self, resp):
        doc_id = resp.id.split('::')[-1] if resp.id else None
        return doc_id
