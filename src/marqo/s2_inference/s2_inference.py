"""This is the interface for interacting with S2 Inference
The functions defined here would have endpoints, later on.
"""
import numpy as np
from marqo.s2_inference.errors import VectoriseError, InvalidModelPropertiesError, ModelLoadError, UnknownModelError, ModelNotInCache
from PIL import UnidentifiedImageError
from marqo.s2_inference.model_registry import load_model_properties
from marqo.s2_inference.configs import get_default_device, get_default_normalization, get_default_seq_length
from marqo.s2_inference.types import *
from marqo.s2_inference.logger import get_logger
import torch

logger = get_logger(__name__)

available_models = dict()
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
        vectorised = available_models[model_cache_key].encode(content, normalize=normalize_embeddings, **kwargs)
    except UnidentifiedImageError as e:
        raise VectoriseError from e

    return _convert_vectorized_output(vectorised)


def _create_model_cache_key(model_name: str, device: str, model_properties: dict = None) -> str:
    """creates a key to store the loaded model by in the cache

    Args:
        model_name (str): _description_
        model_properties (dict): _description_
        device (str): _description_

    Returns:
        str: _description_
    """
    model_cache_key = (model_name, device)

    return model_cache_key


def _update_available_models(model_cache_key: str, model_name: str, validated_model_properties: dict,
                             device: str,
                             normalize_embeddings: bool) -> None:
    """loads the model if it is not already loaded
    """
    if model_cache_key not in available_models:
        try:
            available_models[model_cache_key] = _load_model(model_name,
                                                            validated_model_properties, device=device)
            logger.info(f'loaded {model_name} on device {device} with normalization={normalize_embeddings}')
        except:
            raise ModelLoadError(
                f"Unable to load model={model_name} on device={device} with normalization={normalize_embeddings}. "
                f"If you are trying to load a custom model, "
                f"please check that model_properties={validated_model_properties} is correct "
                f"and the model has valid access permission. ")


def _validate_model_properties(model_name: str, model_properties: dict) -> dict:
    """validate model_properties, if not given then return model_registry properties
    """
    if model_properties is not None:
        """checks model dict to see if all required keys are present
        """
        required_keys = ["name", "dimensions"]
        for key in required_keys:
            if key not in model_properties:
                raise InvalidModelPropertiesError(f"model_properties has missing key '{key}'. ")

        """updates model dict with default values if optional keys are missing
        """
        optional_keys_values = [("type", "sbert"), ("tokens", get_default_seq_length())]
        for key, value in optional_keys_values:
            if key not in model_properties:
                model_properties[key] = value

    else:
        model_properties = get_model_properties_from_registry(model_name)

    return model_properties


def clear_loaded_models() -> None:
    """ clears the loaded model cache

        Future_Change:
            expose cache related functions to the client
    """
    available_models.clear()


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


def _load_model(model_name: str, model_properties: dict, device: str = get_default_device()) -> Any:
    """_summary_

    Args:
        model_name (str): Actual model_name to be fetched from external library
                        prefer passing it in the form of model_properties['name']
        device (str, optional): _description_. Defaults to 'cpu'.

    Returns:
        Any: _description_
    """
    print(f"loading for: model_name={model_name} and properties={model_properties}")
    loader = _get_model_loader(model_properties['name'], model_properties)

    max_sequence_length = model_properties.get('tokens', get_default_seq_length())

    model = loader(model_properties['name'], device=device, embedding_dim=model_properties['dimensions'],
                   max_seq_length=max_sequence_length)

    model.load()

    return model


def get_available_models():
    return available_models


def eject_model(model_name:str,device:str):
    model_cache_key = _create_model_cache_key(model_name, device)
    if model_cache_key in available_models:
        del available_models[model_cache_key]
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return {"result": "success", "message": f"successfully eject model_name \`{model_name}\` from device \`{device}\`"}
    else:
        raise ModelNotInCache(f"The model_name \`{model_name}\` device \`{device}\` is not cached")

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
