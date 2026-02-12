from marqo_common import build_model_properties
from marqo.core.inference.api.exceptions import UnsupportedModelError
from marqo.settings.settings import get_settings


settings = get_settings()


def get_model_properties(model_name: str, marqo_default_models_s3_bucket: str = settings.marqo_default_models_s3_bucket) -> dict:
    try:
        return build_model_properties(model_name, marqo_default_models_s3_bucket)
    except KeyError:
        raise UnsupportedModelError(f"The specified model '{model_name}' is not supported ")


def validate_model_properties(properties: dict) -> None:
    """This is just a very shallow validation as the detailed validation is done in the vectorise call"""
    required_fields = ["dimensions", "type"]

    for field in required_fields:
        if field not in properties:
            raise ValueError(f"Model properties must include '{field}'.")