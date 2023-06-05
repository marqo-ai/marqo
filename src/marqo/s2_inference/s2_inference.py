"""This is the interface for interacting with S2 Inference
The functions defined here would have endpoints, later on.
"""
import functools
import time

import numpy as np
from marqo.errors import ModelCacheManagementError
from marqo.s2_inference.errors import (VectoriseError, InvalidModelPropertiesError, ModelLoadError,
                                       UnknownModelError, ModelNotInCacheError)
from PIL import UnidentifiedImageError
from marqo.s2_inference.model_registry import load_model_properties
from marqo.s2_inference.configs import get_default_device, get_default_normalization, get_default_seq_length
from marqo.s2_inference.types import *
from marqo.s2_inference.logger import get_logger
import torch
import datetime
from marqo.s2_inference import constants
from marqo.tensor_search.utils import read_env_vars_and_defaults
from marqo.tensor_search.enums import AvailableModelsKey
from marqo.tensor_search.configs import EnvVars
import threading
from marqo.tensor_search.utils import read_env_vars_and_defaults, generate_batches
from marqo.tensor_search.configs import EnvVars
from marqo.errors import ConfigurationError

logger = get_logger(__name__)

# The avaiable has the structure:
# {"model_cache_key_1":{"model" : model_object, "most_recently_used_time": time, "model_size" : model_size}}
available_models = dict()
# A lock to protect the model loading process
lock = threading.Lock()
MODEL_PROPERTIES = load_model_properties()


def vectorise(model_name: str, content: Union[str, List[str]], model_properties: dict = None,
              device: str = get_default_device(), normalize_embeddings: bool = get_default_normalization(),
              **kwargs) -> List[List[float]]:
    """vectorizes the content by model name

    Args:
        model_name (str) : Acts as an identifying alias if model_properties is given.
                        If model_properties is None then model_name is used to fetch properties from model_registry
        content (_type_): _description_
        model_properties(dict): {"name": str, "dimensions": int, "tokens": int, "type": str}
                                if model_properties['name'] is not in model_registry, these properties are used to fetch the model
                                if model_properties['name'] is in model_registry, default properties are overridden
                                model_properties can be None only if model_name is a model present in the registry

    Returns:
        List[List[float]]: _description_

    Raises:
        VectoriseError: if the content can't be vectorised, for some reason.
    """

    validated_model_properties = _validate_model_properties(model_name, model_properties)
    model_cache_key = _create_model_cache_key(model_name, device, validated_model_properties)

    _update_available_models(model_cache_key, model_name, validated_model_properties, device, normalize_embeddings)

    try:
        if isinstance(content, str):
            vectorised = available_models[model_cache_key][AvailableModelsKey.model].encode(content, normalize=normalize_embeddings, **kwargs)
        else:
            vector_batches = []
            batch_size = _get_max_vectorise_batch_size()
            for batch in generate_batches(content, batch_size=batch_size):
                vector_batches.append(_convert_tensor_to_numpy(available_models[model_cache_key][AvailableModelsKey.model].encode(batch, normalize=normalize_embeddings, **kwargs)))
            if not vector_batches or all(
                    len(batch) == 0 for batch in vector_batches):  # Check for empty vector_batches or empty arrays
                raise RuntimeError(f"Vectorise created an empty list of batches! Content: {content}")
            else:
                vectorised = np.concatenate(vector_batches, axis=0)
    except UnidentifiedImageError as e:
        raise VectoriseError(str(e)) from e

    return _convert_vectorized_output(vectorised)


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
        raise ConfigurationError(value_error_msg)
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
                             device: str,
                             normalize_embeddings: bool) -> None:
    """loads the model if it is not already loaded.
    Note this method assume the model_properties are validated.
    """
    if model_cache_key not in available_models:
        model_size = get_model_size(model_name, validated_model_properties)
        if lock.locked():
            raise ModelCacheManagementError("Request rejected, as this request attempted to update the model cache, while"
                                            "another request was updating the model cache at the same time.\n"
                                            "Please wait for 10 seconds and send the request again.\n"
                                            "Marqo's documentation can be found here: `https://docs.marqo.ai/latest/`")
        with lock:
            _validate_model_into_device(model_name, validated_model_properties, device,
                                       calling_func=_update_available_models.__name__)
            try:
                most_recently_used_time = datetime.datetime.now()
                available_models[model_cache_key] = {AvailableModelsKey.model: _load_model(model_name,
                                                                                           validated_model_properties,
                                                                                           device=device,
                                                                                           calling_func = _update_available_models.__name__),
                                                     AvailableModelsKey.most_recently_used_time: most_recently_used_time,
                                                     AvailableModelsKey.model_size: model_size}
                logger.info(
                    f'loaded {model_name} on device {device} with normalization={normalize_embeddings} at time={most_recently_used_time}.')
            except Exception as e:
                logger.error(f"Error loading model {model_name} on device {device} with normalization={normalize_embeddings}. \n"
                             f"Error message is {str(e)}")
                raise ModelLoadError(
                    f"Unable to load model={model_name} on device={device} with normalization={normalize_embeddings}. "
                    f"If you are trying to load a custom model, "
                    f"please check that model_properties={validated_model_properties} is correct "
                    f"and Marqo has access to the weights file.")
    else:
        most_recently_used_time = datetime.datetime.now()
        logger.debug(f'renew {model_name} on device {device} with new time={most_recently_used_time}.')
        try:
            available_models[model_cache_key][AvailableModelsKey.most_recently_used_time] = most_recently_used_time
        except KeyError:
            raise ModelNotInCacheError(f"Marqo cannot renew model {model_name} on device {device} with normalization={normalize_embeddings}. "
                                       f"Maybe another thread is updating the model cache at the same time."
                                       f"Please wait for 10 seconds and send the request again.\n")


def _validate_model_properties(model_name: str, model_properties: dict) -> dict:
    """validate model_properties, if not given then return model_registry properties
    """
    if model_properties is not None:
        """checks model dict to see if all required keys are present
        """
        if model_properties.get("type", None) in (None, "sbert"):
            required_keys = ["name", "dimensions"]
            for key in required_keys:
                if key not in model_properties:
                    raise InvalidModelPropertiesError(f"model_properties has missing key '{key}'."
                                                      f"please update your model properties with required key `{key}`"
                                                      f"check `https://docs.marqo.ai/0.0.12/Models-Reference/dense_retrieval/` for more info.")

            """updates model dict with default values if optional keys are missing
            """
            optional_keys_values = [("type", "sbert"), ("tokens", get_default_seq_length())]
            for key, value in optional_keys_values:
                if key not in model_properties:
                    model_properties[key] = value

        elif model_properties.get("type", None) in ("clip", "open_clip"):
            required_keys = ["name", "dimensions"]
            for key in required_keys:
                if key not in model_properties:
                    raise InvalidModelPropertiesError(f"model_properties has missing key '{key}'."
                                                      f"please update your model properties with required key `{key}`"
                                                      f"check `https://docs.marqo.ai/0.0.12/Models-Reference/dense_retrieval/` for more info.")
    else:
        model_properties = get_model_properties_from_registry(model_name)

    return model_properties


def _validate_model_into_device(model_name:str, model_properties: dict, device: str, calling_func: str = None) -> bool:
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
    if _check_memory_threshold_for_model(device, model_size, calling_func = _validate_model_into_device.__name__):
        return True
    else:
        model_cache_key_for_device = [key for key in list(available_models) if key.endswith(device)]
        sorted_key_for_device = sorted(model_cache_key_for_device,
                                       key=lambda x: available_models[x][
                                           AvailableModelsKey.most_recently_used_time])
        for key in sorted_key_for_device:
            logger.info(
                f"Eject model = `{key.split('||')[0]}` with size = `{available_models[key].get('model_size', constants.DEFAULT_MODEL_SIZE)}` from device = `{device}` "
                f"to save space for model = `{model_name}`.")
            del available_models[key]
            if _check_memory_threshold_for_model(device, model_size, calling_func = _validate_model_into_device.__name__):
                return True

        if _check_memory_threshold_for_model(device, model_size, calling_func = _validate_model_into_device.__name__) is False:
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
        used_memory = sum([available_models[key].get("model_size", constants.DEFAULT_MODEL_SIZE) for key, values in
                           available_models.items() if key.endswith(device)])
        threshold = float(read_env_vars_and_defaults(EnvVars.MARQO_MAX_CUDA_MODEL_MEMORY))
    elif device.startswith("cpu"):
        used_memory = sum([available_models[key].get("model_size", constants.DEFAULT_MODEL_SIZE) for key, values in
                           available_models.items() if key.endswith("cpu")])
        threshold = float(read_env_vars_and_defaults(EnvVars.MARQO_MAX_CPU_MODEL_MEMORY))
    else:
        raise ModelCacheManagementError(
            f"Unable to check the device cache for device=`{device}`. The model loading will proceed"
            f"without device cache check. This might break down Marqo if too many models are loaded.")
    if model_size > threshold:
        raise ModelCacheManagementError(
            f"You are trying to load a model with size = `{model_size}` into device = `{device}`, which is larger than the device threshlod = `{threshold}`."
            f"Marqo CANNOT find enough space for the model. Please change the threshold by adjusting the environment variables.\n"
            f"You can find more detailed information at `https://docs.marqo.ai/0.0.17/Advanced-Usage/configuration/`.")
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


def _load_model(model_name: str, model_properties: dict, device: Optional[str] = None, calling_func: str = None) -> Any:
    """_summary_

    Args:
        model_name (str): Actual model_name to be fetched from external library
                        prefer passing it in the form of model_properties['name']
        device (str, optional): _description_. Defaults to 'cpu'.

    Returns:
        Any: _description_
    """
    if calling_func not in ["unit_test", "_update_available_models"]:
        raise RuntimeError(f"The function `{_load_model.__name__}` should only be called by "
                           f"`unit_test` or `_update_available_models` for threading safeness.")

    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - loading for: model_name={model_name} and properties={model_properties}")
    if device is None: device = get_default_device()
    loader = _get_model_loader(model_properties['name'], model_properties)

    max_sequence_length = model_properties.get('tokens', get_default_seq_length())

    model = loader(model_properties['name'], device=device, embedding_dim=model_properties['dimensions'],
                   max_seq_length=max_sequence_length, model_properties=model_properties)

    model.load()

    return model


def clear_loaded_models() -> None:
    """ clears the loaded model cache

        Future_Change:
            expose cache related functions to the client
    """
    available_models.clear()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def get_model_properties_from_registry(model_name: str) -> dict:
    """ Returns a dict describing properties of a model.

    These properties will be used by the tensor_search application to set up
    index parameters.

    see https://huggingface.co/sentence-transformers for available models

    TODO: standardise these dicts

    Returns:
        dict: a dictionary describing properties of the model.
    """

    if model_name not in MODEL_PROPERTIES['models']:
        raise UnknownModelError(f"Could not find model properties in model registry for model={model_name}. "
                                f"Model is not supported by default.")

    return MODEL_PROPERTIES['models'][model_name]


def _check_output_type(output: List[List[float]]) -> bool:
    """checks the output type conforms to what we want

    Args:
        output (List[List[float]]): _description_

    Returns:
        bool: _description_
    """

    if not isinstance(output, list):
        return False
    elif len(output) == 0:
        raise ValueError("received empty input")

    if not isinstance(output[0], list):
        return False
    elif len(output[0]) == 0:
        raise ValueError("received empty input")

    # this is a soft check for speed reasons
    if not isinstance(output[0][0], (float, int)):
        return False

    return True


def _float_tensor_to_list(output: FloatTensor, device: str = get_default_device()) -> Union[
    List[List[float]], List[float]]:
    """

    Args:
        output (FloatTensor): _description_

    Returns:
        List[List[float]]: _description_
    """

    return output.detach().to(device).tolist()


def _nd_array_to_list(output: ndarray) -> Union[List[List[float]], List[float]]:
    """

    Args:
        output (ndarray): _description_

    Returns:
        List[List[float]]: _description_
    """

    return output.tolist()


def _convert_tensor_to_numpy(output:Union[FloatTensor, Tensor]) -> ndarray:
    """
    A function that convert tensors to numpy arrays
    """
    if isinstance(output, (torch.Tensor, torch.FloatTensor)):
        return output.to('cpu').detach().numpy()
    elif isinstance(output, ndarray):
        return output
    else:
        raise ValueError(f"Marqo received an unexpected output type=`{type(output).__name__}`from encode function.")


def _convert_vectorized_output(output: Union[FloatTensor, ndarray, List[List[float]]], fp16: bool = False) -> List[
    List[float]]:
    """converts the model outputs to a list of lists of floats
    also checks the input dim, expects (samples x vector dim)
    if a single sample is present, will pad the first dim to make it (1 x vector_dim)

    Args:
        output (FloatTensor): _description_

    Returns:
        List[List[float]]: _description_
    """

    if _check_output_type(output):
        return output

    if isinstance(output, (FloatTensor, Tensor)):
        if output.ndim == 1:
            output = output.unsqueeze(0)
        output = _float_tensor_to_list(output)

    elif isinstance(output, ndarray):
        if output.ndim == 1:
            output = output[np.newaxis, :]

        output = _nd_array_to_list(output)

    elif isinstance(output, list):
        if isinstance(output[0], FloatTensor):
            output = [_float_tensor_to_list(_o) for _o in output]
        elif isinstance(output[0], ndarray):
            output = [_nd_array_to_list(_o) for _o in output]
        else:
            raise TypeError(f"unsupported nested list with elements of type {type(output[0])}")

    else:
        raise TypeError(f"unsupported output type of {type(output)}")

    if fp16:
        output = np.array(output).astype(np.float16).tolist()

    if _check_output_type(output):
        return output

    raise TypeError(f"unable to convert input of type {type(output)} to a list of lists of floats")


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


def get_available_models():
    return available_models


def eject_model(model_name: str, device: str):
    model_cache_keys = available_models.keys()

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
        raise ModelNotInCacheError(f"The model_name `{model_name}` device `{device}` is not cached or found")

    if model_cache_key in available_models:
        del available_models[model_cache_key]
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return {"result": "success", "message": f"successfully eject model_name `{model_name}` from device `{device}`"}
    else:
        raise ModelNotInCacheError(f"The model_name `{model_name}` device `{device}` is not cached or found")

# def normalize(inputs):

#     is_valid = False
#     if isinstance(inputs, FloatTensor):
#         n_dims = inputs.dim()
#         if n_dims == 2:
#             row_sums = inputs.norm(dim=-1, keepdim=True)
#             is_valid = True
#     elif isinstance(inputs, ndarray):
#         n_dims = inputs.ndim
#         if n_dims == 2:
#             row_sums = np.linalg.norm(inputs, axis=1, ord=2)[:, np.newaxis]
#             is_valid = True
#     elif isinstance(inputs, list):
#         return normalize(np.array(inputs))
#     else:
#         raise TypeError(f"unrecognized type {type(inputs)}")

#     if is_valid:
#         return inputs / row_sums

#     raise TypeError(f"expected 2D matrix for normalization but received {n_dims}")
