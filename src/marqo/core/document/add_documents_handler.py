import json
import uuid
from abc import ABC, abstractmethod
from timeit import default_timer as timer
from typing import List, Dict, Optional, Any

# TODO replace these exception with core exception
from marqo.api.exceptions import InvalidDocumentIdError, InvalidArgError, DocTooLargeError, MarqoWebError

from marqo.config import Config
from marqo.core.document.models.add_docs_params import AddDocsParams
from marqo.core.models import MarqoIndex
from marqo.core.models.marqo_add_documents_response import MarqoAddDocumentsItem, MarqoAddDocumentsResponse
from marqo.tensor_search import utils, enums
from marqo.vespa.models.get_document_response import Document


class AddDocumentsHandler(ABC):

    def __init__(self, marqo_index: MarqoIndex, config: Config, add_docs_params: AddDocsParams):
        self.marqo_index = marqo_index
        self.add_docs_params = add_docs_params
        self.config = config

    def add_documents(self):
        """
        Template method for adding documents to Marqo index
        """
        responses: List[Optional[MarqoAddDocumentsItem]] = [None] * len(self.add_docs_params.docs)
        marqo_docs: Dict[int, Dict[str, Any]] = dict()
        visited_doc_ids: Dict[str, int] = dict()

        t0 = timer()
        # Reverse the order to handle duplicate doc override correctly
        for i, doc in enumerate(reversed(self.add_docs_params.docs)):
            loc = len(self.add_docs_params.docs) - 1 - i
            doc_id = None
            marqo_doc = dict()
            try:
                self.validate_doc(doc, loc)
                doc_id = self._validate_doc_id(doc, loc, visited_doc_ids)
                marqo_doc['_id'] = doc_id or str(uuid.uuid4())

                for field_name, field_content in doc.items():
                    # This collects the doc
                    self.handle_field(marqo_doc, field_name, field_content)

                self.handle_multi_modal_fields(marqo_doc['_id'], marqo_doc)

                # we only add doc to marqo_docs if it is valid
                marqo_docs[loc] = marqo_doc

            except MarqoWebError as err:
                # TODO check different exceptions
                responses[loc] = MarqoAddDocumentsItem.from_error(doc_id, err)

        # retrieve existing docs for existing tensor
        if self.add_docs_params.use_existing_tensors:
            result = self.config.vespa_client.get_batch(list(visited_doc_ids.keys()), self.marqo_index.schema_name)
            existing_vespa_docs = {r.id: r.document for r in result.responses if r.status == 200}
            self.handle_existing_tensors(existing_vespa_docs)

        # vectorise tensor fields
        doc_id_location_map = {doc['_id']: loc for loc, doc in marqo_docs.items()}
        for invalid_doc_item in self.vectorise_tensor_fields():
            loc = doc_id_location_map[invalid_doc_item.id]
            # TODO remove generated doc_id
            responses[loc] = invalid_doc_item
            del marqo_docs[loc]

        # persist to vespa if there are still valid docs
        if marqo_docs:
            for add_doc_item in self.persist_to_vespa(marqo_docs):
                loc = doc_id_location_map[add_doc_item.id]
                # TODO remove generated doc_id
                responses[loc] = add_doc_item

        return MarqoAddDocumentsResponse(errors=False, index_name=self.marqo_index.name, items=responses,
                                         processingTimeMs=((timer() - t0) * 1000))

    @abstractmethod
    def handle_field(self, marqo_doc, field_name, field_content):
        pass

    @abstractmethod
    def handle_multi_modal_fields(self, doc_id: str, marqo_doc: Dict[str, Any]):
        pass

    @abstractmethod
    def handle_existing_tensors(self, existing_vespa_docs: Dict[str, Document]):
        pass

    @abstractmethod
    def vectorise_tensor_fields(self) -> List[MarqoAddDocumentsItem]:
        pass

    @abstractmethod
    def persist_to_vespa(self, marqo_docs) -> List[MarqoAddDocumentsItem]:
        pass

    def validate_doc(self, doc, loc: int):
        if not isinstance(doc, dict):
            raise InvalidArgError("Docs must be dicts")

        if len(doc) <= 0:
            raise InvalidArgError("Can't index an empty dict.")

        max_doc_size = utils.read_env_vars_and_defaults(var=enums.EnvVars.MARQO_MAX_DOC_BYTES)
        if max_doc_size is not None:
            try:
                serialized = json.dumps(doc)
            except TypeError as e:
                raise InvalidArgError(f"Unable to index document: it is not serializable! "
                                      f"Document: `{doc}` ")
            if len(serialized) > int(max_doc_size):
                maybe_id = f" _id:`{doc['_id']}`" if '_id' in doc else ''
                raise DocTooLargeError(
                    f"Document{maybe_id} with length `{len(serialized)}` exceeds "
                    f"the allowed document size limit of [{max_doc_size}]."
                )

    def _validate_doc_id(self, doc, loc: int, visited_doc_ids: Dict[str, int]) -> Optional[str]:
        if '_id' not in doc:
            return None

        doc_id = doc['_id']
        if not isinstance(doc_id, str):
            raise InvalidDocumentIdError(
                f"Document _id must be a string type! "
                f"Received _id {doc_id} of type `{type(doc_id).__name__}`")

        if doc_id in visited_doc_ids:
            # TODO find a better error type
            raise InvalidDocumentIdError(f"Document will be ignored since doc with the same id"
                                         f" `{doc_id}` at location [{visited_doc_ids[doc_id]}] supersedes this one")
        else:
            # TODO this matches the current behaviour, check if it's better to do all validation first
            visited_doc_ids[doc_id] = loc

        del doc['_id']
        return doc_id
