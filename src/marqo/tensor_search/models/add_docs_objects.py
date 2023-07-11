from pydantic.dataclasses import dataclass
from pydantic import Field, validator
from typing import Optional, Union, Any, Sequence
import numpy as np
from marqo.tensor_search.models.private_models import ModelAuth
from typing import List
from marqo.tensor_search.utils import read_env_vars_and_defaults
from marqo.tensor_search.enums import EnvVars
from marqo.errors import InternalError
from pydantic import BaseModel
from marqo.tensor_search.utils import get_best_available_device
from typing import List, Dict


class AddDocsParamsConfig:
    arbitrary_types_allowed = True


class AddDocsBodyParamsOld(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    documents: Union[Sequence[Union[dict, Any]], np.ndarray]

class AddDocsBodyParamsNew(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    non_tensor_fields: List = Field(default_factory=list)
    use_existing_tensors: bool = False
    image_download_headers: dict = Field(default_factory=dict)
    model_auth: Optional[ModelAuth] = None
    mappings: Optional[dict] = None
    documents: Union[Sequence[Union[dict, Any]], np.ndarray]

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

    """

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    # this should only accept Sequences of dicts, but currently validation lies elsewhere
    docs: Union[Sequence[Union[dict, Any]], np.ndarray]

    index_name: str
    auto_refresh: bool
    device: Optional[str]
    non_tensor_fields: List = Field(default_factory=list)
    image_download_thread_count: int = 20
    image_download_headers: dict = Field(default_factory=dict)
    use_existing_tensors: bool = False
    mappings: Optional[dict] = None
    model_auth: Optional[ModelAuth] = None

    def __init__(self, **data: dict):
        # Ensure `None` and passing nothing are treated the same for device
        if "device" not in data or data["device"] is None:
            data["device"] = get_best_available_device()
        super().__init__(**data)
