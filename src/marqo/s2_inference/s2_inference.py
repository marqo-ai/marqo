"""This is the interface for interacting with S2 Inference
The functions defined here would have endpoints, later on.
"""
import numpy as np

from marqo.s2_inference.model_registry import load_model_properties
from marqo.s2_inference.configs import get_default_device,get_default_normalization,get_default_seq_length
from marqo.s2_inference.types import *
from marqo.s2_inference.logger import get_logger
logger = get_logger(__name__)

available_models = dict()
MODEL_PROPERTIES = load_model_properties()

def vectorise(model_name: str, content: Union[str, List[str]], device: str = get_default_device(), 
                                    normalize_embeddings: bool = get_default_normalization(), **kwargs) -> List[List[float]]:
    """vectorizes the content by model name

    Args:
        model_name (str): _description_
        content (_type_): _description_

    Returns:
        List[List[float]]: _description_
    """
    # TODO map by (model_name, device) instead of model_name
    if model_name not in available_models:
        available_models[model_name] = _load_model(model_name, device=device)
        logger.info(f'loaded {model_name} on device {device} with normalization={normalize_embeddings}')
    vectorised = available_models[model_name].encode(content, normalize=normalize_embeddings, **kwargs)
    return _convert_vectorized_output(vectorised)


def clear_loaded_models() -> None:
    """ clears the loaded model cache
    """
    available_models.clear()


def get_model_properties(model_name: str) -> dict:
    """ Returns a dict describing properties of a model.

    These properties will be used by the Neural_Search application to set up
    index parameters.

    see https://huggingface.co/sentence-transformers for available models

    TODO: standardise these dicts

    Returns:
        dict: a dictionary describing properties of the model.
    """

    if model_name not in MODEL_PROPERTIES['models']:
        raise KeyError(f"model_name={model_name} not in allowed model names")

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


def _float_tensor_to_list(output: FloatTensor, device: str = get_default_device()) -> Union[List[List[float]], List[float]]:
    """

    Args:
        output (FloatTensor): _description_

    Returns:
        List[List[float]]: _description_
    """

    return output.detach().to(device).tolist()


def _nd_array_to_list(output : ndarray) -> Union[List[List[float]], List[float]]:
    """

    Args:
        output (ndarray): _description_

    Returns:
        List[List[float]]: _description_
    """

    return output.tolist()


def _convert_vectorized_output(output: Union[FloatTensor, ndarray, List[List[float]]], fp16: bool = False) -> List[List[float]]:
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


def _get_model_loader(model_name: str) -> Any:
    """ Returns a dict describing properties of a model.

    These properties will be used by the Neural_Search application to set up
    index parameters.

    see https://huggingface.co/sentence-transformers for available models

    TODO: standardise these dicts

    Returns:
        dict: a dictionary describing properties of the model.
    """

    if model_name not in MODEL_PROPERTIES['models']:
        raise KeyError(f"model_name={model_name} not in allowed model names")

    model_type = MODEL_PROPERTIES['models'][model_name]['type']

    if model_type not in MODEL_PROPERTIES['loaders']:
        raise KeyError(f"model_name={model_name} for model_type={model_type} not in allowed model types")

    return MODEL_PROPERTIES['loaders'][model_type]


def _load_model(model_name: str, device: str = get_default_device()) -> Any:
    """_summary_

    Args:
        model_name (str): _description_
        device (str, optional): _description_. Defaults to 'cpu'.

    Returns:
        Any: _description_
    """

    model_properties = get_model_properties(model_name)
    
    loader = _get_model_loader(model_name)
    
    max_sequence_length = model_properties.get('tokens', get_default_seq_length())

    model = loader(model_properties['name'], device=device, embedding_dim=model_properties['dimensions'], 
                    max_seq_length=max_sequence_length)
    
    model.load()

    return model

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
    
