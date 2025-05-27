"""This is the interface for interacting with S2 Inference
The functions defined here would have endpoints, later on.
"""
import datetime
import random
import threading
import time
from typing import List, Dict, Optional
from urllib3.exceptions import ReadTimeoutError

import numpy as np
import torch
from PIL import UnidentifiedImageError
from PIL.Image import Image
from torchvision.transforms import Compose

from marqo import marqo_docs
from marqo.api.configs import EnvVars
from marqo.api.exceptions import ModelCacheManagementError, ConfigurationError, InternalError
from marqo.inference.inference_cache.marqo_inference_cache import MarqoInferenceCache
from marqo.s2_inference import constants
from marqo.s2_inference.configs import get_default_normalization, get_default_seq_length
from marqo.s2_inference.errors import (
    VectoriseError, InvalidModelPropertiesError, ModelLoadError,
    UnknownModelError, ModelNotInCacheError, ModelDownloadError)
from marqo.logging import get_logger
from marqo.s2_inference.model_registry import load_model_properties
from marqo.s2_inference.models.model_type import ModelType
from marqo.core.inference.modality_utils import *
from marqo.s2_inference.types import *
from marqo.tensor_search.enums import AvailableModelsKey
from marqo.tensor_search.models.preprocessors_model import Preprocessors
from marqo.tensor_search.models.private_models import ModelAuth
from marqo.tensor_search.utils import read_env_vars_and_defaults, generate_batches, read_env_vars_and_defaults_ints

logger = get_logger(__name__)

# The avaiable has the structure:
# {"model_cache_key_1":{"model" : model_object, "most_recently_used_time": time, "model_size" : model_size}}
_available_models = dict()
# A lock to protect the model loading process
lock = threading.Lock()
MODEL_PROPERTIES = load_model_properties()


def validate_model_properties(model_name: str, model_properties: dict) -> dict:
    """validate model_properties, if not given then return model_registry properties.

    This is a rough validation as it only checks the minimum required fields and dimensions values. More indepth
    check should be done when loading the model.

    Raises:
        InvalidModelPropertiesError: if the model_properties are invalid
        UnknownModelError: if the model_name is not in the model registry
    """
    if model_properties is not None:
        """checks model dict to see if all required keys are present
        """
        required_keys = []

        if "type" not in model_properties:
            error_message_postfix = "Marqo is loading the model with default type 'sbert' as the type was not provided."
        else:
            error_message_postfix = ""

        model_type = model_properties.get("type", None)

        if model_type in (None, ModelType.SBERT):
            required_keys = ["dimensions", "name"]
            # updates model dict with default values if optional keys are missing for sbert
            optional_keys_values = [("type", ModelType.SBERT), ("tokens", get_default_seq_length())]
            for key, value in optional_keys_values:
                if key not in model_properties:
                    model_properties[key] = value
        elif model_type in (ModelType.OpenCLIP, ModelType.CLIP):
            required_keys = ["name", "dimensions"]
        elif model_type in (ModelType.HF_MODEL, ModelType.HF_STELLA):
            required_keys = ["dimensions"]
        elif model_type in (ModelType.NO_MODEL,):
            required_keys = ["dimensions"]
            if not model_name == "no_model":
                raise InvalidModelPropertiesError(f"To use the 'no_model' feature, you must provide 'model = no_model' "
                                                  f"and 'type = no_model', but received 'model = {model_name}' and "
                                                  f"'type = {model_type}'.")
        elif model_type in (ModelType.Test, ModelType.Random, ModelType.MultilingualClip, ModelType.FP16_CLIP,
                            ModelType.SBERT_ONNX, ModelType.CLIP_ONNX, ModelType.LanguageBind):
            pass
        else:
            raise InvalidModelPropertiesError(f"Invalid model type. Please check the model type in model_properties. "
                                              f"Supported model types are '{ModelType.SBERT}', '{ModelType.OpenCLIP}', "
                                              f"'{ModelType.CLIP}', '{ModelType.HF_MODEL}', '{ModelType.HF_STELLA}', "
                                              f"'{ModelType.NO_MODEL}', "
                                              f"'{ModelType.Test}', '{ModelType.Random}', "
                                              f"'{ModelType.MultilingualClip}', "
                                              f"'{ModelType.FP16_CLIP}', '{ModelType.SBERT_ONNX}', "
                                              f"'{ModelType.CLIP_ONNX}' ")

        for key in required_keys:
            if key not in model_properties:
                raise InvalidModelPropertiesError(f"model_properties has missing key '{key}'. "
                                                  f"please update your model properties with required key `{key}`. "
                                                  f"{error_message_postfix} "
                                                  f"check {marqo_docs.list_of_models()}, "
                                                  f"{marqo_docs.bring_your_own_model()} for more info")

    else:
        model_properties = get_model_properties_from_registry(model_name)

    _validate_model_properties_dimension(model_properties.get("dimensions", None))

    return model_properties


def _validate_model_properties_dimension(dimensions: Optional[int]) -> None:
    """Validate the dimensions value in model_properties as the dimensions value must be a positive integer.

    Raises:
        InvalidModelPropertiesError: if the dimensions value is invalid
        """
    if dimensions is None or not isinstance(dimensions, int) or dimensions < 1:
        raise InvalidModelPropertiesError(
            f"Invalid model properties: 'dimensions' must be a positive integer, but received {dimensions}.")


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

    model_properties = MODEL_PROPERTIES['models'][model_name]

    validate_model_properties(model_name, model_properties)

    return model_properties


def _float_tensor_to_list(output: FloatTensor) -> Union[
    List[List[float]], List[float]]:
    """
    Args:
        output (FloatTensor): _description_

    Returns:
        List[List[float]]: _description_
    """

    # Hardcoded to CPU always
    return output.detach().to("cpu").tolist()


def _nd_array_to_list(output: ndarray) -> Union[List[List[float]], List[float]]:
    """

    Args:
        output (ndarray): _description_

    Returns:
        List[List[float]]: _description_
    """

    return output.tolist()
