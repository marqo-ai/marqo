import copy
import json
import uuid
from abc import ABC, abstractmethod
from http import HTTPStatus
from timeit import default_timer as timer
from typing import List, Dict, Optional, Any, Tuple, Set

from marqo.config import Config
from marqo.core.constants import MARQO_DOC_ID
from marqo.core.document.models.add_docs_params import AddDocsParams
from marqo.core.models import MarqoIndex
from marqo.core.models.marqo_add_documents_response import MarqoAddDocumentsItem, MarqoAddDocumentsResponse
from marqo.tensor_search import utils, enums, validation
from marqo.vespa.models.get_document_response import Document
from marqo.api import exceptions as errors


ORIGINAL_ID='_original_id'


class AddDocumentsError(Exception):
    status_code: int = int(HTTPStatus.BAD_REQUEST)
    error_code: str = 'invalid_argument'
    error_message: str

    def __init__(self, error_message: str,
                 error_code: str = 'invalid_argument',
                 status_code: int = int(HTTPStatus.BAD_REQUEST)) -> None:
        self.error_code = error_code
        self.error_message = error_message
        self.status_code = int(status_code)


class DuplicateDocumentError(AddDocumentsError):
    pass


class AddDocumentsResponseCollector:
    def __init__(self):
        self.start_time = timer()
        # TODO we ignore the location for now, and will add it if needed in the future
        self.responses: List[Tuple[int, MarqoAddDocumentsItem]] = []
        self.errors = False
        self.marqo_docs: Dict[str, Dict[str, Any]] = dict()
        self.marqo_doc_loc_map: Dict[str, int] = dict()
        self.visited_doc_ids: Set[str] = set()

    def visited(self, doc_id: str) -> bool:
        return doc_id in self.visited_doc_ids

    def collect_marqo_doc(self, loc: int, marqo_doc: Dict[str, Any]):
        doc_id = marqo_doc[MARQO_DOC_ID]
        self.marqo_docs[doc_id] = marqo_doc
        self.marqo_doc_loc_map[doc_id] = loc
        if marqo_doc[ORIGINAL_ID] is not None:
            self.visited_doc_ids.add(marqo_doc[ORIGINAL_ID])

    def collect_error_response(self, doc_id: Optional[str], error: AddDocumentsError, loc: Optional[int] = None):
        if isinstance(error, DuplicateDocumentError):
            # This is the current logic, docs with same id supersedes previous ones defined in the batch
            # TODO change the logic when we need to report duplicates as error in the response
            return

        if not loc and doc_id and doc_id in self.marqo_doc_loc_map:
            loc = self.marqo_doc_loc_map[doc_id]

        if doc_id in self.marqo_docs:
            doc_id = self.marqo_docs.pop(doc_id)[ORIGINAL_ID]

        print(f"Error detected for doc {doc_id} at location [{loc}]: {error.error_message}")

        # Even if the last document is invalid, we should not use previous ones?
        if doc_id:
            self.visited_doc_ids.add(doc_id)

        self.responses.append((loc, MarqoAddDocumentsItem(
            id=doc_id if doc_id is not None else '',
            error=error.error_message,
            message=error.error_message,
            status=error.status_code,
            code=error.error_code
        )))

        self.errors = True

    def collect_successful_response(self, doc_id: Optional[str]):
        loc = self.marqo_doc_loc_map.get(doc_id, None)

        self.responses.append((loc, MarqoAddDocumentsItem(
            id=doc_id if doc_id is not None else '',
            status=200,
        )))

    def to_add_doc_responses(self, index_name: str) -> MarqoAddDocumentsResponse:
        processing_time = (timer() - self.start_time) * 1000
        # since we reversed the doc list to skip duplicate docs, we now need to reverse the response
        sorted_responses = [response for _, response in sorted(self.responses, key=lambda r: r[0], reverse=True)]
        return MarqoAddDocumentsResponse(errors=self.errors, index_name=index_name, items=sorted_responses,
                                         processingTimeMs=processing_time)


class AddDocumentsHandler(ABC):

    def __init__(self, marqo_index: MarqoIndex, config: Config, add_docs_params: AddDocsParams):
        self.marqo_index = marqo_index
        self.add_docs_params = add_docs_params
        self.config = config
        self.add_docs_response_collector = AddDocumentsResponseCollector()

    def add_documents(self):
        """
        Template method for adding documents to Marqo index
        """
        for loc, original_doc in enumerate(reversed(self.add_docs_params.docs)):
            doc = copy.deepcopy(original_doc)
            original_id = None
            try:
                self.validate_doc(doc)

                original_id = self.validate_and_pop_doc_id(doc)
                doc_id = original_id or str(uuid.uuid4())
                marqo_doc = {ORIGINAL_ID: original_id, MARQO_DOC_ID: doc_id}  # keep this info for error report

                for field_name, field_content in doc.items():
                    self.handle_field(marqo_doc, field_name, field_content)

                self.handle_multi_modal_fields(marqo_doc)

                self.add_docs_response_collector.collect_marqo_doc(loc, marqo_doc)
            except AddDocumentsError as err:
                self.add_docs_response_collector.collect_error_response(original_id, err, loc)

        # retrieve existing docs for existing tensor
        if self.add_docs_params.use_existing_tensors:
            result = self.config.vespa_client.get_batch(list(self.add_docs_response_collector.visited_doc_ids),
                                                        self.marqo_index.schema_name)
            existing_vespa_docs = {r.id: r.document for r in result.responses if r.status == 200}
            self.handle_existing_tensors(existing_vespa_docs)

        # vectorise tensor fields
        self.vectorise_tensor_fields()

        # persist to vespa if there are still valid docs
        self.persist_to_vespa()

        return self.add_docs_response_collector.to_add_doc_responses(self.marqo_index.name)

    @abstractmethod
    def handle_field(self, marqo_doc, field_name, field_content):
        pass

    @abstractmethod
    def handle_multi_modal_fields(self, marqo_doc: Dict[str, Any]):
        pass

    @abstractmethod
    def handle_existing_tensors(self, existing_vespa_docs: Dict[str, Document]):
        pass

    @abstractmethod
    def vectorise_tensor_fields(self) -> None:
        pass

    @abstractmethod
    def persist_to_vespa(self) -> None:
        pass

    def validate_doc(self, doc) -> None:
        try:
            validation.validate_doc(doc)
        except (errors.InvalidArgError, errors.DocTooLargeError) as err:
            raise AddDocumentsError(err.message, error_code=err.code, status_code=err.status_code) from err

    def validate_and_pop_doc_id(self, doc) -> Optional[str]:
        if MARQO_DOC_ID not in doc:
            return None

        doc_id = doc.pop(MARQO_DOC_ID)
        try:
            validation.validate_id(doc_id)
        except errors.InvalidDocumentIdError as err:
            raise AddDocumentsError(err.message, error_code=err.code, status_code=err.status_code) from err

        if self.add_docs_response_collector.visited(doc_id):
            raise DuplicateDocumentError(f"Document will be ignored since doc with the same id"
                                         f" `{doc_id}` supersedes this one")

        return doc_id

