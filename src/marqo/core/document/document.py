from typing import Dict, List, Union
from timeit import default_timer as timer

from marqo.vespa.models.delete_document_response import DeleteAllDocumentsResponse
from marqo.vespa.vespa_client import VespaClient
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.exceptions import IndexNotFoundError, UnsupportedFeatureError, ParsingError
from marqo.vespa.models import UpdateBatchResponse, VespaDocument
from marqo.core.vespa_index import for_marqo_index as vespa_index_factory
from marqo.core.models.marqo_index import IndexType
import marqo.api.exceptions as api_exceptions
from marqo.core.models.update_documents import UpdateDocumentsParams


class Document:
    """A class that handles the document API in Marqo"""

    def __init__(self, vespa_client: VespaClient, index_management: IndexManagement):
        self.vespa_client = vespa_client
        self.index_management = index_management

    def delete_all_docs_by_index_name(self, index_name: str) -> int:
        """Deletes all documents in the given index by index name"""
        if not self.index_management.index_exists(index_name):
            raise IndexNotFoundError(f"Index {index_name} does not exist")

        marqo_index = self.index_management.get_index(index_name)
        return self.delete_all_docs(marqo_index)

    def delete_all_docs(self, marqo_index):
        """Deletes all documents in the given index by marqo_index object"""
        res: DeleteAllDocumentsResponse = self.vespa_client.delete_all_docs(marqo_index.schema_name)
        return res.document_count

    def partial_update_documents_by_index_name(self, index_name,
                                               update_documents_params: UpdateDocumentsParams) -> Dict:
        if not self.index_management.index_exists(index_name):
            raise IndexNotFoundError(f"Index {index_name} does not exist")

        marqo_index = self.index_management.get_index(index_name)
        return self.partial_update_documents(update_documents_params, marqo_index)

    def partial_update_documents(self, update_documents_params: UpdateDocumentsParams, marqo_index):
        """Partially updates documents in the given index by marqo_index object.

        This is not supported for unstructured indexes."""
        if marqo_index.type == IndexType.Unstructured:
            raise UnsupportedFeatureError("'Partial document update' is not supported for unstructured indexes. "
                                          "Please use 'add_documents' with 'use_existing_tensor=True' instead.")

        start_time = timer()
        vespa_index = vespa_index_factory(marqo_index)
        vespa_documents: List[VespaDocument] = []
        unsuccessful_docs: List = []

        for index, doc in enumerate(update_documents_params.documents):
            try:
                vespa_document = VespaDocument(**vespa_index.to_vespa_partial_document(doc))
                vespa_documents.append(vespa_document)
            except ParsingError as e:
                unsuccessful_docs.append(
                    (index, {'_id': doc.get('_id', ''), 'error': e.message,
                             "status": int(api_exceptions.InvalidArgError.status_code),
                             'code': api_exceptions.InvalidArgError.code})
                )

        vespa_res: UpdateBatchResponse = self.vespa_client.update_batch(vespa_documents, marqo_index.schema_name)

        res = self._translate_update_document_response(vespa_res, unsuccessful_docs,
                                                       marqo_index.name, start_time)

        return res

    def _translate_update_document_response(self, responses: UpdateBatchResponse, unsuccessful_docs: List,
                                            index_name: str, start_time) \
            -> Dict[str, Union[int, List]]:
        """Translates Vespa response dict into Marqo response dict for document update"""
        result_dict = {}
        new_items: List[Dict] = []

        if responses is not None:
            result_dict['errors'] = responses.errors
            for resp in responses.responses:
                id = resp.id.split('::')[-1] if resp.id else None
                new_items.append({'status': resp.status})
                if id:
                    new_items[-1].update({'_id': id})
                if resp.message:
                    if resp.status >= 400:
                        new_items[-1].update({'error': resp.message})
                    else:
                        new_items[-1].update({'message': resp.message})

        if unsuccessful_docs:
            result_dict['errors'] = True

        for loc, error_info in unsuccessful_docs:
            new_items.insert(loc, error_info)

        result_dict["index_name"] = index_name
        result_dict["items"] = new_items
        result_dict["processingTimeMs"] = (timer() - start_time) * 1000
        return result_dict
