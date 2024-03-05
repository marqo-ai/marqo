from timeit import default_timer as timer
from typing import Dict, List, Tuple

import marqo.api.exceptions as api_exceptions
from marqo.core.exceptions import UnsupportedFeatureError, ParsingError
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.models.marqo_index import IndexType
from marqo.core.models.marqo_update_documents_response import MarqoUpdateDocumentsResponse, MarqoUpdateDocumentsItem
from marqo.core.vespa_index import for_marqo_index as vespa_index_factory
from marqo.vespa.models import UpdateBatchResponse, VespaDocument
from marqo.vespa.models.delete_document_response import DeleteAllDocumentsResponse
from marqo.vespa.vespa_client import VespaClient


class Document:
    """A class that handles the document API in Marqo"""

    def __init__(self, vespa_client: VespaClient, index_management: IndexManagement):
        self.vespa_client = vespa_client
        self.index_management = index_management

    def delete_all_docs_by_index_name(self, index_name: str) -> int:
        """Delete all documents in the given index by index name

        Args:
            index_name: The name of the index to delete documents from"""
        marqo_index = self.index_management.get_index(index_name)
        return self.delete_all_docs(marqo_index)

    def delete_all_docs(self, marqo_index) -> int:
        """Deletes all documents in the given index by marqo_index object

        Args:
            marqo_index: The index object to delete documents from"""
        res: DeleteAllDocumentsResponse = self.vespa_client.delete_all_docs(marqo_index.schema_name)
        return res.document_count

    def partial_update_documents_by_index_name(self, index_name,
                                               partial_documents: List[Dict]) \
            -> Dict:
        """Partially updates documents in the given index by index name.

        Args:
            index_name: The name of the index to partially update documents in
            partial_documents: A list of documents to partially update

        Raises:
            IndexNotFoundError: If the index does not exist

        Return:
            A dict containing the response of the partial update operation
        """
        marqo_index = self.index_management.get_index(index_name)

        res: MarqoUpdateDocumentsResponse = self.partial_update_documents(partial_documents, marqo_index)
        return res.dict(exclude_none=True)

    def partial_update_documents(self, partial_documents: List[Dict], marqo_index) \
            -> MarqoUpdateDocumentsResponse:
        """Partially updates documents in the given index by marqo_index object.

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

        for index, doc in enumerate(partial_documents):
            try:
                vespa_document = VespaDocument(**vespa_index.to_vespa_partial_document(doc))
                vespa_documents.append(vespa_document)
            except ParsingError as e:
                unsuccessful_docs.append(
                    (index, MarqoUpdateDocumentsItem(id=doc.get('_id', ''), error=e.message,
                                                     status=int(api_exceptions.InvalidArgError.status_code))))

        vespa_res: UpdateBatchResponse = self.vespa_client.update_documents_batch(vespa_documents,
                                                                                  marqo_index.schema_name)

        return self._translate_update_document_response(vespa_res, unsuccessful_docs,
                                                        marqo_index.name, start_time)

    def _translate_update_document_response(self, responses: UpdateBatchResponse, unsuccessful_docs: List,
                                            index_name: str, start_time) \
            -> MarqoUpdateDocumentsResponse:
        """Translates Vespa response dict into MarqoUpdateDocumentsResponse for document update

        Args:
            responses: The response from Vespa
            unsuccessful_docs: The list of unsuccessful documents
            index_name: The name of the index
            start_time: The start time of the operation

        Return:
            MarqoUpdateDocumentsResponse containing the response of the partial update operation
        """

        new_items: List[MarqoUpdateDocumentsItem] = []

        if responses is not None:
            for resp in responses.responses:
                doc_id = resp.id.split('::')[-1] if resp.id else None
                item = MarqoUpdateDocumentsItem(
                    _id=doc_id, status=resp.status, message=resp.message
                )
                new_items.append(item)

        for loc, error_info in unsuccessful_docs:
            new_items.insert(loc, error_info)

        return MarqoUpdateDocumentsResponse(indexName=index_name, items=new_items,
                                            preprocessingTime=(timer() - start_time) * 1000)