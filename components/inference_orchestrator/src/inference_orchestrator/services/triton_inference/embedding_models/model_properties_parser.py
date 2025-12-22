from inference_orchestrator.services.errors import InvalidModelPropertiesError
from inference_orchestrator.services.triton_inference.embedding_models import (
    HuggingFaceModel,
    OpenCLIPModel,
    RandomModel,
)


def get_model_loader(model_properties: dict):
    """
    Parse the model properties and return the appropriate model class.
    """
    model_type = model_properties.get("type")
    if model_type == "hf":
        return HuggingFaceModel
    elif model_type == "open_clip":
        return OpenCLIPModel
    elif model_type == "random":
        return RandomModel
    else:
        raise InvalidModelPropertiesError(f"Unsupported model type: {model_type}")
