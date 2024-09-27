from typing import List
from typing import Optional, Union, Any, Sequence

import numpy as np
from pydantic import BaseModel, validator, root_validator
from pydantic import Field

from marqo import marqo_docs
from marqo.api.exceptions import BadRequestError
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.utils import get_best_available_device, read_env_vars_and_defaults_ints
from marqo.tensor_search.enums import EnvVars


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
    imageDownloadThreadCount: int = Field(default_factory=lambda: read_env_vars_and_defaults_ints(EnvVars.MARQO_IMAGE_DOWNLOAD_THREAD_COUNT_PER_REQUEST))
    mediaDownloadThreadCount: Optional[int]
    textChunkPrefix: Optional[str] = None

    @root_validator
    def validate_thread_counts(cls, values):
        image_count = values.get('imageDownloadThreadCount')
        media_count = values.get('mediaDownloadThreadCount')
        if media_count is not None and image_count != read_env_vars_and_defaults_ints(EnvVars.MARQO_IMAGE_DOWNLOAD_THREAD_COUNT_PER_REQUEST):
            raise ValueError("Cannot set both imageDownloadThreadCount and mediaDownloadThreadCount")
        return values
