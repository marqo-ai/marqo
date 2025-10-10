from typing import Union

from inference_orchestrator.services.errors import InvalidModelPropertiesError
from inference_orchestrator.services.triton_inference.embedding_models import HuggingFaceModelProperties, \
    OpenCLIPModelProperties, HuggingFaceModel, OpenCLIPModel, RandomModelProperties, RandomModel


def parse_model_properties(model_properties: dict) -> Union[HuggingFaceModelProperties, OpenCLIPModelProperties, RandomModelProperties]:
    """Parse the model properties and return the appropriate model properties object.

    Args:
        model_properties (dict): The model properties to parse.

    Returns:
        Union[HuggingFaceModelProperties, OpenCLIPModelProperties, RandomModelProperties]: The parsed model properties object.

    Raises:
        ValueError: If the model type is not supported.
    """
    model_type = model_properties.get("type")
    if model_type == "hf":
        return HuggingFaceModelProperties(**model_properties)
    elif model_type == "open_clip":
        return OpenCLIPModelProperties(**model_properties)
    elif model_type == "random":
        return RandomModelProperties(**model_properties)
    else:
        raise InvalidModelPropertiesError(f"Unsupported model type: {model_type}")


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