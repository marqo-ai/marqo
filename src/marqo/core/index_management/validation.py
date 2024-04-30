from marqo.tensor_search.models.index_settings import IndexSettings
from pydantic import ValidationError
import marqo.logging

logger = marqo.logging.get_logger(__name__)


def validate_settings_object(index_name: str, settings_dict: dict) -> None:
    """
    Validates index settings using the IndexSettings model.

    Args:
        index_name (str): The name of the index to validate settings for.
        settings_dict (dict): A dictionary of settings to validate.

    Raises:
        ValidationError: If the settings do not conform to the IndexSettings model.
        Exception: If an unexpected error occurs during the validation process.

    Returns:
        None: If the validation is successful, nothing is returned.
    """
    try:
        index_settings = IndexSettings(**settings_dict)
        index_settings.to_marqo_index_request(index_name)
    except ValidationError as e:
        logger.debug(f'Validation error for index {index_name}: {e}')
        raise
    except Exception as e:
        logger.error(f'Exception while validating index {index_name}: {e}')
        raise
