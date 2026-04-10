from marqo_common import build_model_properties

from inference_orchestrator.core.settings import get_settings
from inference_orchestrator.services.errors import UnsupportedModelError

settings = get_settings()


def get_model_properties(
    model_name: str,
    marqo_default_models_s3_bucket: str = settings.marqo_default_models_s3_bucket,
) -> dict:
    try:
        return build_model_properties(model_name, marqo_default_models_s3_bucket)
    except KeyError:
        raise UnsupportedModelError(
            f"The specified model '{model_name}' is not supported "
        )
