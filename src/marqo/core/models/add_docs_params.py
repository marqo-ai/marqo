from typing import List
from typing import Optional, Union, Any, Sequence

import numpy as np
from pydantic import BaseModel, validator, root_validator
from pydantic import Field

from marqo import marqo_docs
from marqo.api.exceptions import BadRequestError
from marqo.tensor_search.enums import EnvVars
# TODO move deps
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.utils import read_env_vars_and_defaults_ints


class AddDocsParams(BaseModel):
    """Represents the parameters of the document.add_documents() function

    Params:
        index_name: name of the index
        docs: List of documents
        use_existing_tensors: Whether to use the vectors already in doc (for update docs)
        device: Device used to carry out the document update, if `None` is given, it will be determined inference
        image_download_thread_count: number of threads used to concurrently download images
        media_download_headers: headers to authenticate media download requests
        mappings: a dictionary used to handle all the object field content in the doc,
            e.g., multimodal_combination field
        model_auth: an object used to authorise downloading an object from a datastore
        text_chunk_prefix: an optional prefix to add to each text chunk
    """

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    # this should only accept Sequences of dicts, but currently validation lies elsewhere
    docs: Union[Sequence[Union[dict, Any]], np.ndarray]

    index_name: str
    device: Optional[str]
    tensor_fields: Optional[List] = Field(default_factory=None)
    image_download_thread_count: int = Field(default_factory=lambda: read_env_vars_and_defaults_ints(
        EnvVars.MARQO_IMAGE_DOWNLOAD_THREAD_COUNT_PER_REQUEST))
    media_download_thread_count: Optional[int]
    media_download_headers: Optional[dict] = None
    use_existing_tensors: bool = False
    mappings: Optional[dict] = None
    model_auth: Optional[ModelAuth] = None
    text_chunk_prefix: Optional[str] = None

    def __init__(self, **data: Any):
        super().__init__(**data)

    @root_validator
    def validate_thread_counts(cls, values):
        image_count = values.get('image_download_thread_count')
        media_count = values.get('media_download_thread_count')
        if media_count is not None and image_count != read_env_vars_and_defaults_ints(EnvVars.MARQO_IMAGE_DOWNLOAD_THREAD_COUNT_PER_REQUEST):
            raise ValueError("Cannot set both image_download_thread_count and media_download_thread_count")
        return values

    @validator('docs')
    def validate_docs(cls, docs):
        doc_count = len(docs)

        max_doc = read_env_vars_and_defaults_ints(EnvVars.MARQO_MAX_DOCUMENTS_BATCH_SIZE)

        if doc_count == 0:
            raise BadRequestError(message="Received empty add documents request")
        elif doc_count > max_doc:
            raise BadRequestError(
                message=f"Number of docs in add documents request ({doc_count}) exceeds limit of {max_doc}. "
                        f"If using the Python client, break up your `add_documents` request into smaller batches using "
                        f"its `client_batch_size` parameter. "
                        f"See {marqo_docs.api_reference_document_body()} for more details."
            )

        return docs