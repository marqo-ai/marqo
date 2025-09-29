from marqo.inference.triton_inference.embedding_models.hugging_face.hugging_face_model_properties import HuggingFaceModelProperties
from marqo.inference.triton_inference.embedding_models.open_clip.open_clip_model_properties import OpenCLIPModelProperties
from typing import Union


def parse_model_properties(model_properties: dict) -> Union[HuggingFaceModelProperties, OpenCLIPModelProperties]:
    """Parse the model properties and return the appropriate model properties object.

    Args:
        model_properties (dict): The model properties to parse.

    Returns:
        Union[HuggingFaceModelProperties, OpenCLIPModelProperties]: The parsed model properties object.

    Raises:
        ValueError: If the model type is not supported.
    """
    model_type = model_properties.get("type")
    if model_type == "hugging_face":
        return HuggingFaceModelProperties(**model_properties),
    elif model_type == "open_clip":
        return OpenCLIPModelProperties(**model_properties)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")