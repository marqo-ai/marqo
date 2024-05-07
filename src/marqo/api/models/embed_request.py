"""Classes used for API communication

Choices (enum-type structure) in fastAPI:
https://pydantic-docs.helpmanual.io/usage/types/#enums-and-choices
"""
import pydantic
from typing import Union, List, Dict, Optional, Any

from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.models.api_models import BaseMarqoModel
from marqo.core.embed.embed import EmbedContentType



class EmbedRequest(BaseMarqoModel):
    # content can be a single query or list of queries. Queries can be a string or a dictionary.
    content: Union[str, Dict[str, float], List[Union[str, Dict[str, float]]]]
    image_download_headers: Optional[Dict] = None
    modelAuth: Optional[ModelAuth] = None
    content_type: Optional[EmbedContentType] = EmbedContentType.Query

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