from typing import List
from typing import Optional, Union, Any, Sequence

import numpy as np
from pydantic import BaseModel, root_validator
from pydantic import Field

from marqo.core.models.add_docs_params import BatchVectorisationMode
from marqo.tensor_search.enums import EnvVars
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.utils import read_env_vars_and_defaults_ints


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
    # This parameter is experimental for now. we will add it to the document and py-marqo once it has been verified
    batchVectorisationMode: BatchVectorisationMode = BatchVectorisationMode.PER_DOCUMENT

    @root_validator
    def validate_thread_counts(cls, values):
        image_count = values.get('imageDownloadThreadCount')
        media_count = values.get('mediaDownloadThreadCount')
        if media_count is not None and image_count != read_env_vars_and_defaults_ints(EnvVars.MARQO_IMAGE_DOWNLOAD_THREAD_COUNT_PER_REQUEST):
            raise ValueError("Cannot set both imageDownloadThreadCount and mediaDownloadThreadCount")
        return values
