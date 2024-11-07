from typing import List, Dict
from typing import Optional, Any, Sequence

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
    mediaDownloadHeaders: Optional[dict] = None
    modelAuth: Optional[ModelAuth] = None
    mappings: Optional[dict] = None
    documents: Sequence[Dict[str, Any]]
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

    @root_validator(skip_on_failure=True)
    def _validate_image_download_headers_and_media_download_headers(cls, values):
        """Validate imageDownloadHeaders and mediaDownloadHeaders. Raise an error if both are set.

        If imageDownloadHeaders is set, set mediaDownloadHeaders to it and use mediaDownloadHeaders in the
        rest of the code.

        imageDownloadHeaders is deprecated and will be removed in the future.
        """
        image_download_headers = values.get('imageDownloadHeaders')
        media_download_headers = values.get('mediaDownloadHeaders')
        if image_download_headers and media_download_headers:
            raise ValueError("Cannot set both imageDownloadHeaders and mediaDownloadHeaders. "
                             "'imageDownloadHeaders' is deprecated and will be removed in the future. "
                             "Use mediaDownloadHeaders instead.")
        if image_download_headers:
            values['mediaDownloadHeaders'] = image_download_headers
        return values
