from enum import Enum
from decimal import Decimal
from typing import Optional, Tuple
from marqo.tensor_search.models.index_settings import IndexSettings
from pydantic import ValidationError
from fastapi.responses import JSONResponse
import json


def validate_settings_object(index_name, settings_json) -> Tuple[int, Optional[str]]:
    """
    Validates index settings.

    Returns:
        A tuple containing an HTTP status code and optionally a dictionary or an error message. 
        On success, it returns 200 and a dictionary representing the settings.
        On failure due to an ValidationError, it returns 400 and an error message.
        For any other exception, it returns 500 and an error message.
    """
    try:
        settings_object = json.loads(settings_json)
        index_settings = IndexSettings(**settings_object)
        marqo_request = index_settings.to_marqo_index_request(index_name)
        # Convert the successful marqo_request (if needed) to a dictionary representation
        convert_marqo_request_to_dict(marqo_request)
        return JSONResponse(
            content={
                "validated": True,
                "index": index_name
            },
            status_code=200
        )
    except ValidationError as e:
        return JSONResponse(
            content={
                "validated": False,
                "validation_error": str(e),
                "index": index_name
            },
            status_code=400
        )
    except Exception as e:
        return JSONResponse(
            content={
                "validated": False,
                "validation_error": str(e),
                "index": index_name
            },
            status_code=500
        )


def convert_marqo_request_to_dict(index_settings):
    """Converts a MarqoIndexRequest to a dictionary.
    Returns
        A dictionary representation of the MarqoIndexRequest
    """
    index_settings_dict = index_settings.dict(exclude_none=True)
    return convert_enums_to_values(index_settings_dict)


def convert_enums_to_values(data):
    if isinstance(data, dict):
        return {key: convert_enums_to_values(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_enums_to_values(element) for element in data]
    elif isinstance(data, Enum):
        if isinstance(data.value, float):
            return Decimal(str(data.value))
        return data.value
    elif isinstance(data, float):
        return Decimal(str(data))
    else:
        return data
