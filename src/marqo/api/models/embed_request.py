"""Classes used for API communication

Choices (enum-type structure) in fastAPI:
https://pydantic-docs.helpmanual.io/usage/types/#enums-and-choices
"""
from typing import Union, List, Dict, Optional

import pydantic
from pydantic import Field, field_validator, model_validator

from marqo.base_model import StrictBaseModel, ImmutableBaseModel
from marqo.core.embed.embed import EmbedContentType
from marqo.tensor_search.models.private_models import ModelAuth


class EmbedRequest(StrictBaseModel):
    # content can be a single query or list of queries. Queries can be a string or a dictionary.
    content: Union[str, List[str], Dict[str, List[str]]]
    imageDownloadHeaders: Optional[Dict] = Field(default=None, alias="image_download_headers")
    mediaDownloadHeaders: Optional[Dict] = None
    model_name: str = Field(alias="modelName", default="")
    model_properties: Optional[Dict] = Field(alias="modelProperties", default=None)
    model_auth: Optional[ModelAuth] = Field(alias="modelAuth", default=None)
    normalize_embeddings: bool = Field(alias="normalizeEmbeddings", default=True)
    return_dimensions: Optional[bool] = Field(alias="returnDimensions", default=False)
    use_cuda: Optional[bool] = Field(alias="useCuda", default=False)
    content_type: Optional[EmbedContentType] = Field(default=EmbedContentType.Query, alias="contentType")

    @field_validator('content')
    def validate_non_empty_content(cls, v):
        """
        For the case when content is a string, make sure it's not empty. For the case when content is a list,
        make sure it's not empty. For the case when content is a dict, make sure there's at least one key.
        """
        if isinstance(v, str) and v.strip() == "":
            raise ValueError(f"content has no text when stripped: {v}")
        elif isinstance(v, list) and len(v) == 0:
            raise ValueError("empty content list detected.")
        elif isinstance(v, dict) and len(v) == 0:
            raise ValueError("empty content dictionary detected.")
        return v

    @model_validator(mode='after')
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