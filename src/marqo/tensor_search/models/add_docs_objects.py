from pydantic.dataclasses import dataclass
from pydantic import Field
from typing import Optional, Union, Any, Sequence
import numpy as np
from marqo.tensor_search.models.private_models import ModelAuth
from typing import List
from marqo.errors import InternalError
from marqo.tensor_search.utils import read_env_vars_and_defaults
from marqo.tensor_search.enums import EnvVars
from marqo.errors import InternalError


class AddDocsParamsConfig:
    arbitrary_types_allowed = True

def _get_default_device() -> str:
    '''Return the default device for AddDocsParams.device.
    An InternalError will be raised if the default device cannot be determined.'''
    device = read_env_vars_and_defaults().get(EnvVars.MARQO_BEST_AVAILABLE_DEVICE)
    if not device:
        raise InternalError("Best available device was not properly determined on Marqo startup.")
    return device

@dataclass(frozen=True, config=AddDocsParamsConfig)
class AddDocsParams:
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

    # this should only accept Sequences of dicts, but currently validation lies elsewhere
    docs: Union[Sequence[Union[dict, Any]], np.ndarray]

    index_name: str
    auto_refresh: bool
    non_tensor_fields: List = Field(default_factory=list)
    device: str = Field(if_none=_get_default_device)
    image_download_thread_count: int = 20
    image_download_headers: dict = Field(default_factory=dict)
    use_existing_tensors: bool = False
    mappings: Optional[dict] = None
    model_auth: Optional[ModelAuth] = None
