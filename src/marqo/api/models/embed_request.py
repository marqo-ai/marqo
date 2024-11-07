"""Classes used for API communication

Choices (enum-type structure) in fastAPI:
https://pydantic-docs.helpmanual.io/usage/types/#enums-and-choices
"""
from typing import Union, List, Dict, Optional

import pydantic
from pydantic import Field, root_validator

from marqo.base_model import MarqoBaseModel
from marqo.core.embed.embed import EmbedContentType
from marqo.tensor_search.models.private_models import ModelAuth


class EmbedRequest(MarqoBaseModel):
    # content can be a single query or list of queries. Queries can be a string or a dictionary.
    content: Union[str, Dict[str, float], List[Union[str, Dict[str, float]]]]
    imageDownloadHeaders: Optional[Dict] = Field(default=None, alias="image_download_headers")
    mediaDownloadHeaders: Optional[Dict] = None
    modelAuth: Optional[ModelAuth] = None
    content_type: Optional[EmbedContentType] = Field(default=EmbedContentType.Query, alias="contentType")

    @pydantic.validator('content')
    def validate_content(cls, value):
        # Iterate through content list items
        if (isinstance(value, list) or isinstance(value, dict)) and len(value) == 0:
            raise ValueError("Embed content list should not be empty")

        # Convert all types of content into a list
        if isinstance(value, str) or isinstance(value, dict):
            list_to_validate = [value]
        elif isinstance(value, List):
            list_to_validate = value
        else:
            raise ValueError("Embed content should be a string, a dictionary, or a list of strings or dictionaries")

        for item in list_to_validate:
            if isinstance(item, str):
                continue
            elif isinstance(item, dict):
                if len(item) == 0:
                    raise ValueError("Dictionary content should not be empty")
                for key in item:
                    if not isinstance(key, str):
                        raise ValueError("Keys in dictionary content should all be strings")
                    if not isinstance(item[key], float):
                        raise ValueError("Values in dictionary content should all be floats")
            else:
                raise ValueError("Embed content should be a string, a dictionary, or a list of strings or dictionaries")

        return value

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