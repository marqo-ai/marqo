import threading

from marqo.core.inference.api import ModelManager, ModelError
from marqo.inference.triton_inference.embedding_models.abstract_embedding_model import AbstractEmbeddingModel
from marqo.inference.triton_inference.triton.triton_grpc_client import TritonGRPCClient
from marqo.logging import get_logger
from marqo.s2_inference.errors import (
    InvalidModelPropertiesError)
from marqo.s2_inference.model_registry import load_model_properties
from marqo.s2_inference.models.model_type import ModelType
from marqo.s2_inference.types import *
from marqo.tensor_search.enums import AvailableModelsKey
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.inference.triton_inference.embedding_models.open_clip.open_clip_model_properties import OpenCLIPModelProperties
from marqo.inference.triton_inference.embedding_models.open_clip.open_clip_model import OpenCLIPModel
from marqo.inference.triton_inference.embedding_models.hugging_face.hugging_face_model import HuggingFaceModel
from typing import Annotated
from marqo.inference.triton_inference.embedding_models.model_properties_parser import parse_model_properties

logger = get_logger(__name__)
_available_models = dict()
MODEL_PROPERTIES = load_model_properties()
lock = threading.Lock()


def load_model(
        model_name: str, model_properties: dict, triton_client: TritonGRPCClient,
        model_manager: ModelManager, model_auth: Optional[ModelAuth] = None,
    ) -> AbstractEmbeddingModel:
    """
    Load the model and preprocessor if not already loaded
    """
    # TODO.Triton - Remove device
    model_cache_key = _create_model_cache_key(model_name, "cpu", model_properties)
    _update_available_models(
        model_cache_key, model_name, model_properties,
        triton_client=triton_client, model_manager=model_manager, model_auth=model_auth
    )
    model = _available_models[model_cache_key][AvailableModelsKey.model]
    return model


def get_available_models() -> Dict:
    """Returns the available models in the cache."""
    return _available_models


def _create_model_cache_key(model_name: str, device: str, model_properties: dict = None) -> str:
    """creates a key to store the loaded model by in the cache

    Args:
        model_name (str): _description_
        model_properties (dict): _description_
        device (str): _description_

    Returns:
        str: _description_
    """
    # Changing the format of model cache key will also need to change eject_model api

    if model_properties is None:
        model_properties = dict()

    model_cache_key = (
            model_name + "||" +
            model_properties.get('name', '') + "||" +
            str(model_properties.get('dimensions', '')) + "||" +
            model_properties.get('type', '') + "||" +
            str(model_properties.get('tokens', '')) + "||" +
            device
    )

    return model_cache_key


def _update_available_models(model_cache_key: str, model_name: str, validated_model_properties: dict,
                             triton_client: TritonGRPCClient, model_manager: ModelManager,
                             model_auth: ModelAuth = None,) -> None:
    """loads the model if it is not already loaded.
    Note this method assume the model_properties are validated.
    """
    if model_cache_key not in _available_models:
        _available_models[model_cache_key] = {
            AvailableModelsKey.model: _load_model(
                model_name,
                validated_model_properties,
                model_auth=model_auth,
                triton_client=triton_client,
                model_manager=model_manager,
            ),
        }


def _validate_model_properties_dimension(dimensions: Optional[int]) -> None:
    """Validate the dimensions value in model_properties as the dimensions value must be a positive integer.

    Raises:
        InvalidModelPropertiesError: if the dimensions value is invalid
        """
    if dimensions is None or not isinstance(dimensions, int) or dimensions < 1:
        raise InvalidModelPropertiesError(
            f"Invalid model properties: 'dimensions' must be a positive integer, but received {dimensions}.")


def _load_model(
        model_name: str, model_properties: dict,
        model_manager: ModelManager, triton_client: TritonGRPCClient,
        model_auth: Optional[ModelAuth] = None
) -> Any:
    """_summary_

    Args:
        model_name (str): Actual model_name to be fetched from external library
                        prefer passing it in the form of model_properties['name']
        device (str): Required. Should always be passed when loading model
        model_auth: Authorisation details for downloading a model (if required)

    Returns:
        Any: _description_
    """
    model_type = model_properties.get('type')

    if not model_type:
        raise InvalidModelPropertiesError("Model properties must include a 'type' field.")

    model_loader = {
        "hf": HuggingFaceModel,
        "open_clip": OpenCLIPModel,
    }[model_type]

    model = model_loader(
        triton_client=triton_client,
        model_properties=model_properties,
        model_manager=model_manager,
    )
    model.load()
    return model


def clear_loaded_models() -> None:
    """ clears the loaded model cache

        Future_Change:
            expose cache related functions to the client
    """
    _available_models.clear()


def _get_model_loader(model_name: str, model_properties: dict) -> Any:
    """ Returns a dict describing properties of a model.

    These properties will be used by the tensor_search application to set up
    index parameters.

    see https://huggingface.co/sentence-transformers for available models

    TODO: standardise these dicts

    Returns:
        dict: a dictionary describing properties of the model.
    """

    model_type = model_properties['type']

    if model_type not in MODEL_PROPERTIES['loaders']:
        raise KeyError(f"model_name={model_name} for model_type={model_type} not in allowed model types")

    return MODEL_PROPERTIES['loaders'][model_type]


# class TritonModelManager(ModelManager):
#     """
#     A class that handles the communication with the marqo model management container.
#     """
#     def __init__(self):
#         pass
#
#     def get_loaded_models(self) -> dict:
#         """Returns the available models in the cache."""
#         raise NotImplementedError()
#
#     def eject_model(self, model_name: str, device: str) -> dict:
#         model_cache_keys = _available_models.keys()
#
#         model_cache_key = None
#
#         # we can't handle the situation where there are two models with the same name and device
#         # but different properties.
#         for key in model_cache_keys:
#             if isinstance(key, str):
#                 if key.startswith(model_name) and key.endswith(device):
#                     model_cache_key = key
#                     break
#             else:
#                 continue
#
#         if model_cache_key is None:
#             raise ModelError(f"The model_name `{model_name}` device `{device}` is not cached or found")
#
#         if model_cache_key in _available_models:
#             del _available_models[model_cache_key]
#             if device.startswith("cuda"):
#                 torch.cuda.empty_cache()
#             return {"result": "success",
#                     "message": f"successfully eject model_name `{model_name}` from device `{device}`"}
#         else:
#             raise ModelError(f"The model_name `{model_name}` device `{device}` is not cached or found")
#
#     def load_model(self, model_properties: dict):

