import enum
from typing import List
from typing import Optional, Union, Any, Sequence

import numpy as np
from pydantic import BaseModel, validator, root_validator
from pydantic import Field

from marqo import marqo_docs
from marqo.api.exceptions import BadRequestError

# TODO move deps
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.utils import get_best_available_device, read_env_vars_and_defaults_ints
from marqo.tensor_search.enums import EnvVars


class BatchVectorisationMode(enum.Enum):
    PER_FIELD = 'per_field'
    PER_DOCUMENT = 'per_document'
    PER_BATCH = 'per_batch'


class AddDocsParams(BaseModel):
    """Represents the parameters of the tensor_search.add_documents() function

    Params:
        index_name: name of the index
        docs: List of documents
        use_existing_tensors: Whether to use the vectors already in doc (for update docs)
        device: Device used to carry out the document update, if `None` is given, it will be determined by
                EnvVars.MARQO_BEST_AVAILABLE_DEVICE
        image_download_thread_count: number of threads used to concurrently download images
        image_download_headers: headers to authenticate image download
        mappings: a dictionary used to handle all the object field content in the doc,
            e.g., multimodal_combination field
        model_auth: an object used to authorise downloading an object from a datastore
        text_chunk_prefix: an optional prefix to add to each text chunk
        batch_vectorisation_mode: choose how we batch vectorisation requests to the embedding model.
                                  supports per_field, per_document and per_batch [Experimental]
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
    image_download_headers: dict = Field(default_factory=dict)
    use_existing_tensors: bool = False
    mappings: Optional[dict] = None
    model_auth: Optional[ModelAuth] = None
    text_chunk_prefix: Optional[str] = None
    # This parameter is experimental for now. we will add it to the document and py-marqo once it has been verified
    batch_vectorisation_mode: BatchVectorisationMode = BatchVectorisationMode.PER_DOCUMENT

    def __init__(self, **data: Any):
        # Ensure `None` and passing nothing are treated the same for device
        if "device" not in data or data["device"] is None:
            data["device"] = get_best_available_device()
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