from typing import List, Dict
from typing import Optional, Any, Sequence

from pydantic.v1 import BaseModel, root_validator
from pydantic.v1 import Field

from marqo.tensor_search.models.private_models import ModelAuth


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
    imageDownloadThreadCount: Optional[int] = None
    mediaDownloadThreadCount: Optional[int] = None
    textChunkPrefix: Optional[str] = None

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