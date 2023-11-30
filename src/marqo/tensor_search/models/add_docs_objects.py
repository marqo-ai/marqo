from pydantic.dataclasses import dataclass
from pydantic import Field, validator
from typing import Optional, Union, Any, Sequence
import numpy as np
from marqo.tensor_search.models.private_models import ModelAuth
from typing import List
from marqo.tensor_search.utils import read_env_vars_and_defaults
from marqo.tensor_search.enums import EnvVars
from marqo.errors import InternalError, BadRequestError
from pydantic import BaseModel, root_validator, validator
from marqo.tensor_search.utils import get_best_available_device
from marqo.tensor_search import utils
from typing import List, Dict


"""
Validation functions for AddDocsParams fields
"""
def validate_add_docs_count(docs: Union[Sequence[Union[dict, Any]], np.ndarray]):
    """
    Validates that the doc count is within the allowed range.
    docs cannot be empty.
    """

    doc_count = len(docs)
    max_add_docs_count = utils.read_env_vars_and_defaults_ints(EnvVars.MARQO_MAX_ADD_DOCS_COUNT)

    if doc_count == 0:
        raise BadRequestError(message="Received empty add documents request")
    elif doc_count > max_add_docs_count:
        raise BadRequestError(message=f"Number of docs in add documents request ({doc_count}) exceeds limit of {max_add_docs_count}. "
                                    f"This limit can be increased by setting the environment variable `{EnvVars.MARQO_MAX_ADD_DOCS_COUNT}`. "
                                    f"If using the Python client, break up your `add_documents` request into smaller batches using its `client_batch_size` parameter. "
                                    f"See https://docs.marqo.ai/1.3.0/API-Reference/documents/#body for more details.")

    return docs


class AddDocsParamsConfig:
    arbitrary_types_allowed = True


class AddDocsBodyParams(BaseModel):
    """
    Representation of the body parameters of the API add_or_replace_documents() function.
    This will be processed by add_docs_params_orchestrator (along with other query parameters)
    into an AddDocsParams object to be given to tensor_search.add_documents()
    """
    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False
        extra = "forbid" # Raise error on unknown fields

    nonTensorFields: List = None
    tensorFields: List = None
    useExistingTensors: bool = False
    imageDownloadHeaders: dict = Field(default_factory=dict)
    modelAuth: Optional[ModelAuth] = None
    mappings: Optional[dict] = None
    documents: Union[Sequence[Union[dict, Any]], np.ndarray]
    textChunkPrefix: Optional[str] = None


class AddDocsParams(BaseModel):
    """Represents the parameters of the tensor_search.add_documents() function

    Params:
        index_name: name of the index
        docs: List of documents
        auto_refresh: Set to False if indexing lots of docs
        non_tensor_fields: List of fields, within documents to not create tensors for. Default to
          make tensors for all fields.
        use_existing_tensors: Whether to use the vectors already in doc (for update docs)
        device: Device used to carry out the document update, if `None` is given, it will be determined by
                EnvVars.MARQO_BEST_AVAILABLE_DEVICE
        image_download_thread_count: number of threads used to concurrently download images
        image_download_headers: headers to authenticate image download
        mappings: a dictionary used to handle all the object field content in the doc,
            e.g., multimodal_combination field
        model_auth: an object used to authorise downloading an object from a datastore
        text_chunk_prefix: string added to the front of every generated text chunk for vectorisation. Not actually stored as text in the document.

    """

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    # this should only accept Sequences of dicts, but currently validation lies elsewhere
    docs: Union[Sequence[Union[dict, Any]], np.ndarray]

    index_name: str
    auto_refresh: bool
    device: Optional[str]
    non_tensor_fields: Optional[List] = Field(default_factory=list)
    tensor_fields: Optional[List] = Field(default_factory=None)
    image_download_thread_count: int = 20
    image_download_headers: dict = Field(default_factory=dict)
    use_existing_tensors: bool = False
    mappings: Optional[dict] = None
    model_auth: Optional[ModelAuth] = None
    text_chunk_prefix: Optional[str] = None

    @root_validator
    def validate_fields(cls, values):
        """
        Validates that exactly one of `tensor_fields` or `non_tensor_fields` is defined
        Note: InternalError is raised here instead of user error, as user errors will be caught separately
        by the API validation layer.
        """
        
        field1 = values.get('tensor_fields')
        field2 = values.get('non_tensor_fields')

        if field1 is not None and field2 is not None:
            raise InternalError("Only one of `tensor_fields` or `non_tensor_fields` can be provided.")
        if field1 is None and field2 is None:
            raise InternalError("Exactly one of `tensor_fields` or `non_tensor_fields` must be provided.")

        return values
    
    @validator('docs')
    def validate_docs(cls, docs):
        """
        Validates that docs length does not exceed the maximum and is not empty.
        More validation can be added in the future.
        """
        
        docs = validate_add_docs_count(docs)

        return docs
    


    def __init__(self, **data: Any):
        # Ensure `None` and passing nothing are treated the same for device
        if "device" not in data or data["device"] is None:
            data["device"] = get_best_available_device()
        super().__init__(**data)