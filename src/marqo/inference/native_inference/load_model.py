import datetime
import threading
from typing import Dict, Optional

import torch
from torchvision.transforms import Compose

from marqo import marqo_docs
from marqo.api.configs import EnvVars
from marqo.api.exceptions import ModelCacheManagementError, ConfigurationError, InternalError
from marqo.core.inference.api import ModelError, ModelManager
from marqo.inference.native_inference.embedding_models.abstract_embedding_model import AbstractEmbeddingModel
from marqo.s2_inference import constants
from marqo.s2_inference.configs import get_default_normalization, get_default_seq_length
from marqo.s2_inference.errors import (
    InvalidModelPropertiesError, ModelLoadError,
    ModelNotInCacheError, ModelDownloadError)
from marqo.logging import get_logger
from marqo.s2_inference.model_registry import load_model_properties
from marqo.s2_inference.models.model_type import ModelType
from marqo.s2_inference.types import *
from marqo.tensor_search.enums import AvailableModelsKey
from marqo.tensor_search.models.preprocessors_model import Preprocessors
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.utils import read_env_vars_and_defaults

logger = get_logger(__name__)
_available_models = dict()
MODEL_PROPERTIES = load_model_properties()
lock = threading.Lock()


def load_model(model_name: str, model_properties: dict, model_auth: Optional[ModelAuth], device: str) -> AbstractEmbeddingModel:
    """
    Load the model and preprocessor if not already loaded
    """
    validated_model_properties = model_properties
    model_cache_key = _create_model_cache_key(model_name, device, validated_model_properties)


    _update_available_models(
        model_cache_key, model_name, validated_model_properties, device, normalize_embeddings=True, model_auth=model_auth
    )

    model = _available_models[model_cache_key][AvailableModelsKey.model]

    return model


def get_available_models() -> Dict:
    """Returns the available models in the cache."""
    return _available_models


def is_preprocessor_preload(model_properties: dict = None) -> bool:
    """Check if the model should be preloaded with an image preprocessor to preprocess image tensor_search module
        model_properties: Validated model properties. The model properties should have been validated in marqo_index
    """
    model_type = model_properties.get("type", None)
    return model_type in constants.PREPROCESS_PRELOAD_MODELS


def load_multimodal_model_and_get_preprocessors(model_name: str, model_properties: Optional[dict] = None,
                                                device: Optional[str] = None,
                                                model_auth: Optional[ModelAuth] = None,
                                                normalize_embeddings: bool = get_default_normalization()) \
        -> Tuple[Any, Preprocessors]:
    """Load the model and return preprocessors for different modalities.

    Args:
        model_name (str): The name of the model to load.
        model_properties (dict): The validated properties of the model.
        device (str): The device to load the model on.
        model_auth: Authorisation details for downloading a model (if required)
        normalize_embeddings (bool): Whether to normalize the embeddings.

    Returns:
        Tuple[Any, Dict[str, Optional[Compose]]]: The loaded model and a dictionary of preprocessors for different modalities.

    Raises:
        InternalError: If the device is not set.
    """
    if not device:
        raise InternalError(message=f"vectorise (internal function) cannot be called without setting device!")

    # if model_properties.get("type") in ['languagebind', 'imagebind']:
    #    model = load_multimodal_model(model_name, model_properties, device)

    model_cache_key = _create_model_cache_key(model_name, device, model_properties)

    _update_available_models(
        model_cache_key, model_name, model_properties, device, normalize_embeddings,
        model_auth=model_auth
    )

    model = _available_models[model_cache_key][AvailableModelsKey.model]

    if model_properties.get("type") in ['languagebind']:
        preprocessors = model.get_preprocessors()
    elif model_properties.get("type") in [ModelType.OpenCLIP, ModelType.CLIP]:
        preprocessors = {"image": getattr(model, "preprocess", None)}
    else:
        raise InternalError(f"Model type {model_properties.get('type')} does not support preprocessors pre loading in"
                            f"add_document ")
    return model, Preprocessors(**preprocessors)


def _get_max_vectorise_batch_size() -> int:
    """Gets MARQO_MAX_VECTORISE_BATCH_SIZE from the environment, validates it before returning it."""

    max_batch_size_value = read_env_vars_and_defaults(EnvVars.MARQO_MAX_VECTORISE_BATCH_SIZE)
    validation_error_msg = (
        "Could not properly read env var `MARQO_MAX_VECTORISE_BATCH_SIZE`. "
        "`MARQO_MAX_VECTORISE_BATCH_SIZE` must be an int greater than or equal to 1."
    )
    try:
        batch_size = int(max_batch_size_value)
    except (ValueError, TypeError) as e:
        value_error_msg = f"`{validation_error_msg} Current value: `{max_batch_size_value}`. Reason: {e}"
        logger.error(value_error_msg)
        raise ConfigurationError(value_error_msg) from e
    if batch_size < 1:
        batch_size_too_small_msg = f"`{validation_error_msg} Current value: `{max_batch_size_value}`."
        logger.error(batch_size_too_small_msg)
        raise ConfigurationError(batch_size_too_small_msg)
    return batch_size


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

    model_cache_key = (model_name + "||" +
                       model_properties.get('name', '') + "||" +
                       str(model_properties.get('dimensions', '')) + "||" +
                       model_properties.get('type', '') + "||" +
                       str(model_properties.get('tokens', '')) + "||" +
                       device)

    return model_cache_key


def _update_available_models(model_cache_key: str, model_name: str, validated_model_properties: dict,
                             device: str, normalize_embeddings: bool, model_auth: ModelAuth = None) -> None:
    """loads the model if it is not already loaded.
    Note this method assume the model_properties are validated.
    """
    if model_cache_key not in _available_models:
        model_size = get_model_size(model_name, validated_model_properties)
        if lock.locked():
            raise ModelCacheManagementError(
                "Request rejected, as this request attempted to update the model cache, while "
                "another request was updating the model cache at the same time. "
                "Please wait for 10 seconds and send the request again ")
        with lock:
            _validate_model_into_device(model_name, validated_model_properties, device,
                                        calling_func=_update_available_models.__name__)
            try:
                most_recently_used_time = datetime.datetime.now()
                _available_models[model_cache_key] = {
                    AvailableModelsKey.model: _load_model(
                        model_name, validated_model_properties,
                        device=device,
                        calling_func=_update_available_models.__name__,
                        model_auth=model_auth
                    ),
                    AvailableModelsKey.most_recently_used_time: most_recently_used_time,
                    AvailableModelsKey.model_size: model_size
                }
                logger.info(
                    f'loaded {model_name} on device {device} with normalization={normalize_embeddings} at time={most_recently_used_time}.')
            except Exception as e:
                logger.error(
                    f"Error loading model {model_name} on device {device} with normalization={normalize_embeddings}. \n"
                    f"Error message is {str(e)}")

                if isinstance(e, ModelDownloadError):
                    raise e
                raise ModelLoadError(
                    f"Unable to load model={model_name} on device={device} with normalization={normalize_embeddings}. "
                    f"If you are trying to load a custom model, "
                    f"please check that model_properties={validated_model_properties} is correct "
                    f"and Marqo has access to the weights file.") from e

    else:
        most_recently_used_time = datetime.datetime.now()
        logger.debug(f'renewed {model_name} on device {device} with new most recently time={most_recently_used_time}.')
        try:
            _available_models[model_cache_key][AvailableModelsKey.most_recently_used_time] = most_recently_used_time
        except KeyError as e:
            raise ModelNotInCacheError(
                f"Marqo cannot renew model {model_name} on device {device} with normalization={normalize_embeddings}. "
                f"Maybe another thread is updating the model cache at the same time."
                f"Please wait for 10 seconds and send the request again.\n") from e


def _validate_model_properties_dimension(dimensions: Optional[int]) -> None:
    """Validate the dimensions value in model_properties as the dimensions value must be a positive integer.

    Raises:
        InvalidModelPropertiesError: if the dimensions value is invalid
        """
    if dimensions is None or not isinstance(dimensions, int) or dimensions < 1:
        raise InvalidModelPropertiesError(
            f"Invalid model properties: 'dimensions' must be a positive integer, but received {dimensions}.")


def _validate_model_into_device(model_name: str, model_properties: dict, device: str, calling_func: str = None) -> bool:
    '''
    Note: this function should only be called by `_update_available_models` for threading safeness.

    A function to detect if the device have enough memory to load the target model.
    If not, it will try to eject some models to spare the space.
    Args:
        model_name: The name of the model to load
        model_properties: The model properties of the model
        device: The target device to laod the model
    Returns:
        True we have enough space for the model
        Raise an error and return False if we can't find enough space for the model.
    '''
    if calling_func not in ["unit_test", "_update_available_models"]:
        raise RuntimeError("This function should only be called by `update_available_models` or `unit_test` for "
                           "thread safeness.")

    model_size = get_model_size(model_name, model_properties)
    if _check_memory_threshold_for_model(device, model_size, calling_func=_validate_model_into_device.__name__):
        return True
    else:
        model_cache_key_for_device = [key for key in list(_available_models) if key.endswith(device)]
        sorted_key_for_device = sorted(model_cache_key_for_device,
                                       key=lambda x: _available_models[x][
                                           AvailableModelsKey.most_recently_used_time])
        for key in sorted_key_for_device:
            logger.info(
                f"Eject model = `{key.split('||')[0]}` with size = `{_available_models[key].get('model_size', constants.DEFAULT_MODEL_SIZE)}` from device = `{device}` "
                f"to save space for model = `{model_name}`.")
            del _available_models[key]
            if _check_memory_threshold_for_model(device, model_size, calling_func=_validate_model_into_device.__name__):
                return True

        if _check_memory_threshold_for_model(device, model_size,
                                             calling_func=_validate_model_into_device.__name__) is False:
            raise ModelCacheManagementError(
                f"Marqo CANNOT find enough space to load model = `{model_name}` in device = `{device}`.\n"
                f"Marqo tried to eject all the models on this device = `{device}` but still can't find enough space. \n"
                f"Please use a smaller model or increase the memory threshold.")


def _check_memory_threshold_for_model(device: str, model_size: Union[float, int], calling_func: str = None) -> bool:
    '''
    Note: this function should only be called by `_validate_model_into_device` for threading safeness.
    `_validate_model_into_device` is calle by `_update_available_models` which is already thread safe.

    Check the memory usage in the target device and decide whether we can add a new model
    Args:
        device: the target device to check
        model_size: the size of the model to load
    Returns:
        True if we have enough space
        False if we don't have enough space
    '''
    if calling_func not in ["unit_test", "_validate_model_into_device"]:
        raise RuntimeError(f"The function `{_check_memory_threshold_for_model.__name__}` should only be called by "
                           f"`unit_test` or `_validate_model_into_device` for threading safeness.")

    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        used_memory = sum([_available_models[key].get("model_size", constants.DEFAULT_MODEL_SIZE) for key, values in
                           _available_models.items() if key.endswith(device)])
        threshold = float(read_env_vars_and_defaults(EnvVars.MARQO_MAX_CUDA_MODEL_MEMORY))
    elif device.startswith("cpu"):
        used_memory = sum([_available_models[key].get("model_size", constants.DEFAULT_MODEL_SIZE) for key, values in
                           _available_models.items() if key.endswith("cpu")])
        threshold = float(read_env_vars_and_defaults(EnvVars.MARQO_MAX_CPU_MODEL_MEMORY))
    else:
        raise ModelCacheManagementError(
            f"Unable to check the device cache for device=`{device}`. The model loading will proceed"
            f"without device cache check. This might break down Marqo if too many models are loaded.")
    if model_size > threshold:
        raise ModelCacheManagementError(
            f"You are trying to load a model with size = `{model_size}` into device = `{device}`, which is larger than the device threshold = `{threshold}`. "
            f"Marqo CANNOT find enough space for the model. Please change the threshold by adjusting the environment variables.\n"
            f"Please modify the threshold by setting the environment variable `MARQO_MAX_CUDA_MODEL_MEMORY` or `MARQO_MAX_CPU_MODEL_MEMORY`."
            f"You can find more detailed information at {marqo_docs.configuring_marqo()}.")
    return (used_memory + model_size) < threshold


def get_model_size(model_name: str, model_properties: dict) -> (int, float):
    '''
    Return the model size for given model
    Note that the priorities are size_in_properties -> model_name -> model_type -> default size
    '''
    if "model_size" in model_properties:
        return model_properties["model_size"]

    name_info = (model_name + model_properties.get("name", "")).lower().replace("/", "-")
    for name, size in constants.MODEL_NAME_SIZE_MAPPING.items():
        if name in name_info:
            return size

    type = model_properties.get("type", None)
    return constants.MODEL_TYPE_SIZE_MAPPING.get(type, constants.DEFAULT_MODEL_SIZE)


def _load_model(
        model_name: str, model_properties: dict, device: str,
        calling_func: str = None, model_auth: Optional[ModelAuth] = None
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
    if calling_func not in ["unit_test", "_update_available_models"]:
        raise RuntimeError(f"The function `{_load_model.__name__}` should only be called by "
                           f"`unit_test` or `_update_available_models` for threading safeness.")

    print(f"loading for: model_name={model_name} and properties={model_properties}")

    model_type = model_properties.get("type")
    loader = _get_model_loader(model_properties.get('name', None), model_properties)

    # TODO For each refactored model class, add a new elif block here and remove the if block
    #  once we have all models refactored
    if model_type in (
            ModelType.OpenCLIP, ModelType.HF_MODEL, ModelType.HF_STELLA, ModelType.LanguageBind,
            ModelType.Random, ModelType.MultilingualClip
    ):
        model = loader(
            device=device,
            model_properties=model_properties,
            model_auth=model_auth,
        )
    else:
        model = loader(
            model_properties.get('name', None),
            device=device,
            embedding_dim=model_properties['dimensions'],
            model_properties=model_properties,
            model_auth=model_auth,
            max_seq_length=model_properties.get('tokens', get_default_seq_length())
        )
    model.load()
    return model


def clear_loaded_models() -> None:
    """ clears the loaded model cache

        Future_Change:
            expose cache related functions to the client
    """
    _available_models.clear()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


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


class NativeModelManager(ModelManager):
    """
    A class to retrieve all loaded models and eject models by key
    """
    def get_loaded_models(self) -> dict:
        """Returns the available models in the cache."""

        return {"models": [{"model_name": ix.split("||")[0], "model_device": ix.split("||")[-1]}
                           for ix in _available_models if isinstance(ix, str)]}

    def eject_model(self, model_name: str, device: str) -> dict:
        model_cache_keys = _available_models.keys()

        model_cache_key = None

        # we can't handle the situation where there are two models with the same name and device
        # but different properties.
        for key in model_cache_keys:
            if isinstance(key, str):
                if key.startswith(model_name) and key.endswith(device):
                    model_cache_key = key
                    break
            else:
                continue

        if model_cache_key is None:
            raise ModelError(f"The model_name `{model_name}` device `{device}` is not cached or found")

        if model_cache_key in _available_models:
            del _available_models[model_cache_key]
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            return {"result": "success",
                    "message": f"successfully eject model_name `{model_name}` from device `{device}`"}
        else:
            raise ModelError(f"The model_name `{model_name}` device `{device}` is not cached or found")
