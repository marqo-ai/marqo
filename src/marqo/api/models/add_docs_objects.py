from typing import List, Dict, Optional, Any, Sequence

from pydantic import BaseModel, Field, model_validator, ConfigDict

from marqo.tensor_search.models.private_models import ModelAuth


class AddDocsBodyParams(BaseModel):
    """The parameters of the body parameters of tensor_search_add_documents() function"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        extra="forbid"
    )

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

    @model_validator(mode='before')
    def _validate_image_download_headers_and_media_download_headers(cls, values):
        """Validate imageDownloadHeaders and mediaDownloadHeaders. Raise an error if both are set.

        If imageDownloadHeaders is set, set mediaDownloadHeaders to it and use mediaDownloadHeaders in the
        rest of the code.

        imageDownloadHeaders is deprecated and will be removed in the future.
        """
        if isinstance(values, dict):
            image_download_headers = values.get('imageDownloadHeaders')
            media_download_headers = values.get('mediaDownloadHeaders')
            if image_download_headers and media_download_headers:
                raise ValueError("Cannot set both imageDownloadHeaders and mediaDownloadHeaders. "
                                 "'imageDownloadHeaders' is deprecated and will be removed in the future. "
                                 "Use mediaDownloadHeaders instead.")
            if image_download_headers:
                values['mediaDownloadHeaders'] = image_download_headers
        return values