import threading
from contextlib import contextmanager
from typing import Any, Dict, Optional, Union

import orjson
from blake3 import blake3
from tritonclient.grpc import InferenceServerClient as TritonGRPCClient

from inference_orchestrator.core.logging import get_logger
from inference_orchestrator.services.errors import (
    InvalidModelPropertiesError,
    ModelOperationInProgressError,
)
from inference_orchestrator.services.triton_inference.embedding_models import (
    HuggingFaceModel,
    OpenCLIPModel,
    RandomModel,
)
from inference_orchestrator.services.triton_inference.embedding_models.model_properties_parser import (
    get_model_loader,
)
from inference_orchestrator.services.triton_inference.model_manager.model_management_client import (
    ModelManagementClient,
)

logger = get_logger(__name__)
_available_models: Dict[str, Union[OpenCLIPModel, HuggingFaceModel]] = dict()
lock = threading.Lock()


@contextmanager
def _model_op_guard(lock: threading.Lock, timeout: float = 2.0):
    """Try to acquire the lock for model operations. Wait for up to 2 seconds to avoid
    bursts of requests causing immediate failures.

    Raise OperationConflictError if the lock cannot be acquired.
    """
    acquired = lock.acquire(timeout=timeout)
    if not acquired:
        raise ModelOperationInProgressError(
            "Another model load/unload operation is in progress. Please try again later "
        )
    try:
        yield
    finally:
        lock.release()


def load_model(
    model_name: str,
    model_properties: dict,
    triton_client: TritonGRPCClient,
    model_management_client: ModelManagementClient,
    timeout: float = 2.0,
) -> Union[OpenCLIPModel, HuggingFaceModel, RandomModel]:
    """
    Load a model based on the provided model name and properties.
    If the model is already loaded, it retrieves it from the cache.
    """
    model_cache_key = _create_model_cache_key(model_name, model_properties)
    with _model_op_guard(lock, timeout=timeout):
        _update_available_models(
            model_cache_key,
            model_properties,
            triton_client=triton_client,
            model_management_client=model_management_client,
        )
    model = _available_models[model_cache_key]
    return model


def get_available_models() -> Dict:
    """Returns the available models in the cache."""
    return _available_models


def _create_model_cache_key(model_name: str, model_properties: dict) -> str:
    """creates a key to store the loaded model by in the cache

    Args:
        model_name (str): _description_
        model_properties (dict): _description_

    Returns:
        str: _description_
    """
    # Changing the format of model cache key will also need to change eject_model api
    model_properties_serialized = orjson.dumps(
        model_properties, option=orjson.OPT_SORT_KEYS
    )
    model_properties_hash = blake3(model_properties_serialized).hexdigest()[:4]
    model_cache_key = f"{model_name}||{model_properties_hash}"
    return model_cache_key


def _update_available_models(
    model_cache_key: str,
    model_properties: dict,
    triton_client: TritonGRPCClient,
    model_management_client: ModelManagementClient,
) -> None:
    """loads the model if it is not already loaded.
    Note this method assume the model_properties are validated.
    """
    if model_cache_key not in _available_models:
        _available_models[model_cache_key] = _load_model(
            model_properties,
            triton_client=triton_client,
            model_management_client=model_management_client,
        )


def _validate_model_properties_dimension(dimensions: Optional[int]) -> None:
    """Validate the dimensions value in model_properties as the dimensions value must be a positive integer.

    Raises:
        InvalidModelPropertiesError: if the dimensions value is invalid
    """
    if dimensions is None or not isinstance(dimensions, int) or dimensions < 1:
        raise InvalidModelPropertiesError(
            f"Invalid model properties: 'dimensions' must be a positive integer, but received {dimensions}."
        )


def _load_model(
    model_properties: dict,
    triton_client: TritonGRPCClient,
    model_management_client: ModelManagementClient,
) -> Any:
    """
    Loads the model based on the provided properties.


    Args:
        model_properties (dict): _description_
        model_management_client (ModelManagementClient): _description_
        triton_client (TritonGRPCClient): _description_

    Returns:
        Any: _description_
    """
    model_loader = get_model_loader(model_properties)

    model: Union[OpenCLIPModel, HuggingFaceModel] = model_loader(
        model_properties=model_properties,
        model_management_client=model_management_client,
        triton_client=triton_client,
    )

    model.load()
    return model


def clear_loaded_models() -> None:
    """clears the loaded model cache

    Future_Change:
        expose cache related functions to the client
    """
    _available_models.clear()


def get_loaded_models(detailed: bool = False) -> Dict:
    """returns the loaded model cache

    Future_Change:
        expose cache related functions to the client
    """
    result = {"models": []}
    for model_cache_key, model in _available_models.items():
        model_name = model_cache_key
        if detailed:
            result["models"].append(
                {
                    "modelName": model_name,
                    "modelProperties": model.model_properties.model_dump_json(
                        by_alias=True
                    ),
                }
            )
        else:
            result["models"].append({"modelName": model_name})
    return result


def eject_model(model_name: str) -> dict:
    """ejects a model from the loaded model cache

    Args:
        model_name (str): the name of the model to eject, including the properties hash suffix

    Returns:
        dict: result of the ejection operation
    """
    with _model_op_guard(lock, timeout=2.0):
        if model_name in _available_models:
            _available_models[model_name].unload()
            del _available_models[model_name]

    return {"result": "success", "message": f"Model {model_name} ejected successfully."}
