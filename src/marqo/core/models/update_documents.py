from typing import Dict, Sequence, Any

from pydantic import validator

from marqo.api.exceptions import BadRequestError
from marqo.base_model import ImmutableStrictBaseModel
from marqo.tensor_search.enums import EnvVars
from marqo.tensor_search.utils import read_env_vars_and_defaults_ints


class UpdateDocumentsParams(ImmutableStrictBaseModel):
    documents: Sequence[Dict[str, Any]]

    @validator('documents')
    def validate_docs(cls, documents):
        doc_count = len(documents)

        max_doc = read_env_vars_and_defaults_ints(EnvVars.MARQO_MAX_DOCUMENTS_BATCH_SIZE)

        if doc_count == 0:
            raise BadRequestError(message="Received empty add documents request")
        elif doc_count > max_doc:
            raise BadRequestError(
                message=f"Number of docs in update_documents request ({doc_count}) exceeds limit of {max_doc}. "
                        f"If using the Python client, break up your `update_documents` request into smaller batches "
                        f"using its `client_batch_size` parameter. "
            )

        return documents
