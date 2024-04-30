from marqo.tensor_search.models.index_settings import IndexSettings
from pydantic import ValidationError
import json
import marqo.logging

logger = marqo.logging.get_logger(__name__)


def validate_settings_object(index_name, settings_json) -> bool:
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
        index_settings.to_marqo_index_request(index_name)
        return True
    except ValidationError as e:
        logger.debug(f'Validation error for index {index_name}: {e}')
        raise e
    except Exception as e:
        logger.error(f'Exception while validating index {index_name}: {e}')
        raise e
