from marqo.vespa.models.delete_document_response import DeleteAllDocumentResponse
from marqo.vespa.vespa_client import VespaClient
from marqo.core.index_management.index_management import IndexManagement
from marqo.core.exceptions import IndexNotFoundError


class Document:
    """A class that handles the document API in Marqo"""

    def __init__(self, vespa_client: VespaClient, index_management: IndexManagement):
        self.vespa_client = vespa_client
        self.index_management = index_management

    def delete_all_docs(self, index_name: str) -> DeleteAllDocumentResponse:
        """Deletes all documents in the given index"""
        if not self.index_management.index_exists(index_name):
            raise IndexNotFoundError(f"Index {index_name} does not exist")

        return self.vespa_client.delete_all_docs(index_name)