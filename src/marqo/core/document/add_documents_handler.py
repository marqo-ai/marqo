import json
import uuid
from abc import ABC, abstractmethod
from timeit import default_timer as timer
from typing import List, Dict, Optional, Any

# TODO replace these exception with core exception
from marqo.api.exceptions import InvalidDocumentIdError, InvalidArgError, DocTooLargeError


from marqo.config import Config
from marqo.core.document.models.add_docs_params import AddDocsParams
from marqo.core.models import MarqoIndex
from marqo.core.models.marqo_add_documents_response import MarqoAddDocumentsItem, MarqoAddDocumentsResponse
from marqo.tensor_search import utils, enums


class AddDocumentsHandler(ABC):

    def __init__(self, marqo_index: MarqoIndex, config: Config, add_docs_params: AddDocsParams):
        self.marqo_index = marqo_index
        self.add_docs_params = add_docs_params
        self.config = config

    def add_documents(self):
        """
        Template method for adding documents to Marqo index
        """
        self.validate_add_docs_params()

        responses: List[MarqoAddDocumentsItem] = []
        marqo_docs: Dict[int, Dict[str, Any]] = dict()
        visited_doc_ids: Dict[str, int] = dict()

        t0 = timer()
        # Reverse the order to handle duplicate doc override correctly
        for i, doc in enumerate(reversed(self.add_docs_params.docs)):
            loc = len(self.add_docs_params.docs) - 1 - i
            doc_id = None
            marqo_doc = dict()
            try:
                self._validate_doc(doc, loc)
                doc_id = self._validate_doc_id(doc, loc, visited_doc_ids)
                marqo_doc['_id'] = doc_id or str(uuid.uuid4())

                for field_name, field_content in doc.items():
                    self.handle_field(marqo_doc, field_name, field_content)

                marqo_docs[loc] = marqo_doc
            except Exception as err:
                responses[loc] = MarqoAddDocumentsItem.from_error(doc_id, err)

        # retrieve existing docs for existing tensor
        if self.add_docs_params.use_existing_tensors:
            result = self.config.vespa_client.get_batch(list(visited_doc_ids.keys()), self.marqo_index.schema_name)
            existing_vespa_docs = {r.id: r.document for r in result.responses if r.status == 200}
            self.handle_existing_tensors(marqo_docs, existing_vespa_docs)

        # download media files
        media_content_repo = self.download_media_contents(marqo_docs)

        # handle tensors
        self.vectorise_tensor_fields(marqo_docs, media_content_repo)

        # persist to vespa
        for loc, response in self.persist_to_vespa(marqo_docs):
            responses.insert(loc, response)

        t1 = timer()
        return MarqoAddDocumentsResponse(index_name=self.marqo_index.name, items=responses, processingTimeMs=t1-t0)

    @abstractmethod
    def validate_add_docs_params(self):
        pass

    @abstractmethod
    def handle_field(self, marqo_doc, field_name, field_content):
        pass

    @abstractmethod
    def handle_existing_tensors(self, marqo_docs, existing_vespa_docs):
        pass

    @abstractmethod
    def download_media_contents(self, marqo_docs):
        pass

    @abstractmethod
    def vectorise_tensor_fields(self, marqo_docs, media_content_repo):
        pass

    @abstractmethod
    def persist_to_vespa(self, marqo_docs) -> Dict[int, MarqoAddDocumentsItem]:
        pass


    def _validate_doc(self, doc, loc: int):
        if not isinstance(doc, dict):
            raise InvalidArgError("Docs must be dicts")

        if len(doc) <= 0:
            raise InvalidArgError("Can't index an empty dict.")

        max_doc_size = utils.read_env_vars_and_defaults(var=enums.EnvVars.MARQO_MAX_DOC_BYTES)
        if max_doc_size is not None:
            try:
                serialized = json.dumps(doc)
            except TypeError as e:
                raise InvalidArgError(f"Unable to index document at location [{loc}]: it is not serializable! "
                                      f"Document: `{doc}` ")
            if len(serialized) > int(max_doc_size):
                maybe_id = f" _id:`{doc['_id']}`" if '_id' in doc else ''
                raise DocTooLargeError(
                    f"Document at location [{loc}]{maybe_id} with length `{len(serialized)}` exceeds "
                    f"the allowed document size limit of [{max_doc_size}]."
                )

    def _validate_doc_id(self, doc, loc: int, visited_doc_ids: Dict[str, int]) -> Optional[str]:
        if '_id' not in doc:
            return None

        doc_id = doc['_id']
        if not isinstance(doc_id, str):
            raise InvalidDocumentIdError(
                f"Document at location [{loc}] _id must be a string type! "
                f"Received _id {doc_id} of type `{type(doc_id).__name__}`")

        if doc_id in visited_doc_ids:
            # TODO find a better error type
            raise InvalidDocumentIdError(f"Document at location [{loc}] will be ignored since doc with the same id"
                                         f" `{doc_id}` at location [{visited_doc_ids[doc_id]}] supersedes this one")
        else:
            # TODO this matches the current behaviour, check if it's better to do all validation first
            visited_doc_ids[doc_id] = loc

        del doc['_id']
        return doc_id
