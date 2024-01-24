from typing import List
from typing import Optional, Union, Any, Sequence

import numpy as np
from pydantic import BaseModel, validator
from pydantic import Field

from marqo.api.exceptions import BadRequestError
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.utils import get_best_available_device


class AddDocsParamsConfig:
    arbitrary_types_allowed = True


class AddDocsBodyParams(BaseModel):
    """The parameters of the body parameters of tensor_search_add_documents() function"""

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False
        extra = "forbid"  # Raise error on unknown fields

    tensorFields: Optional[List] = None
    useExistingTensors: bool = False
    imageDownloadHeaders: dict = Field(default_factory=dict)
    modelAuth: Optional[ModelAuth] = None
    mappings: Optional[dict] = None
    documents: Union[Sequence[Union[dict, Any]], np.ndarray]
    imageDownloadThreadCount: int = 20


# TODO - Make this configurable
MAX_DOCS = 128


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

    """

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    # this should only accept Sequences of dicts, but currently validation lies elsewhere
    docs: Union[Sequence[Union[dict, Any]], np.ndarray]

    index_name: str
    device: Optional[str]
    tensor_fields: Optional[List] = Field(default_factory=None)
    image_download_thread_count: int = 20
    image_download_headers: dict = Field(default_factory=dict)
    use_existing_tensors: bool = False
    mappings: Optional[dict] = None
    model_auth: Optional[ModelAuth] = None

    def __init__(self, **data: Any):
        # Ensure `None` and passing nothing are treated the same for device
        if "device" not in data or data["device"] is None:
            data["device"] = get_best_available_device()
        super().__init__(**data)

    @validator('docs')
    def validate_docs(cls, docs):
        doc_count = len(docs)

        if doc_count == 0:
            raise BadRequestError(message="Received empty add documents request")
        elif doc_count > MAX_DOCS:
            raise BadRequestError(
                message=f"Number of docs in add documents request ({doc_count}) exceeds limit of {MAX_DOCS}. "
                        f"If using the Python client, break up your `add_documents` request into smaller batches using "
                        f"its `client_batch_size` parameter. "
                        f"See https://marqo.pages.dev/2.0.0/API-Reference/documents/#body for more details."
            )

        return docs
