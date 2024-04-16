"""Classes used for API communication

Choices (enum-type structure) in fastAPI:
https://pydantic-docs.helpmanual.io/usage/types/#enums-and-choices
"""
import pydantic
from pydantic import BaseModel, root_validator, Field
from typing import Union, List, Dict, Optional, Any

from marqo.tensor_search import validation
from marqo.tensor_search.enums import SearchMethod
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.models.score_modifiers_object import ScoreModifier
from marqo.tensor_search.models.search import SearchContext, SearchContextTensor
from marqo.api.exceptions import InvalidArgError
from marqo.core.models.marqo_index import MarqoIndex


class EmbeddingRequest(BaseMarqoModel):
    # content can be a single query or list of queries. Queries can be a string or a dictionary.
    content: Union[Union[str, Dict[str, float]], List[Union[str, Dict[str, float]]]]
    image_download_headers: Optional[Dict] = None
    modelAuth: Optional[ModelAuth] = None

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
            raise ValueError("Embed content should be a string, dictionary, or a list of strings or dictionaries")

        for item in list_to_validate:
            if isinstance(item, str):
                pass
            elif isinstance(item, dict):
                if len(item) == 0:
                    raise ValueError("Dictionary content should not be empty")
                if not all(isinstance(k, str) for k in item.keys()):
                    raise ValueError("Keys in dictionary content should all be strings")
                if not all(isinstance(v, float) for v in item.values()):
                    raise ValueError("Values in dictionary content should all be floats")
            else:
                raise ValueError("Embed content should be a string, dictionary, or a list of strings or dictionaries")

        return value