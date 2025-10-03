import threading
from typing import Any, Dict, Optional, Union

from tritonclient.grpc import InferenceServerClient as TritonGRPCClient

from marqo_inference_container.core.logging import get_logger
from marqo_inference_container.services.errors import InvalidModelPropertiesError, ModelOperationInProgressError
from marqo_inference_container.services.triton_inference.embedding_models import OpenCLIPModelProperties, \
    OpenCLIPModel, HuggingFaceModel
from marqo_inference_container.services.triton_inference.embedding_models.model_properties_parser import \
    get_model_loader
from marqo_inference_container.services.triton_inference.model_manager.model_management_client import ModelManagementClient
from contextlib import contextmanager


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
        model_name: str, model_properties: dict, triton_client: TritonGRPCClient, model_management_client: ModelManagementClient
) -> Union[OpenCLIPModel, OpenCLIPModelProperties]:
    """
    Load a model based on the provided model name and properties.
    If the model is already loaded, it retrieves it from the cache.
    """
    model_cache_key = _create_model_cache_key(model_name, model_properties)
    with _model_op_guard(lock):
        _update_available_models(
            model_cache_key, model_name, model_properties,
            triton_client=triton_client, model_management_client=model_management_client
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
    model_cache_key = (
            model_name + "||" +
            model_properties.get('name', '') + "||" +
            str(model_properties.get('dimensions', '')) + "||" +
            model_properties.get('type', '') + "||" +
            str(model_properties.get('tokens', '')) + "||"
    )

    return model_cache_key


def _update_available_models(model_cache_key: str, model_name: str, model_properties: dict,
                             triton_client: TritonGRPCClient, model_management_client: ModelManagementClient) -> None:
    """loads the model if it is not already loaded.
    Note this method assume the model_properties are validated.
    """
    if model_cache_key not in _available_models:
        _available_models[model_cache_key] = _load_model(
            model_name,
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
            f"Invalid model properties: 'dimensions' must be a positive integer, but received {dimensions}.")


def _load_model(
        model_name: str, model_properties: dict, triton_client: TritonGRPCClient, model_management_client: ModelManagementClient,
) -> Any:
    """_summary_

    Args:
        model_name (str): Actual model_name to be fetched from external library
                        prefer passing it in the form of model_properties['name']
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
    """ clears the loaded model cache

        Future_Change:
            expose cache related functions to the client
    """
    _available_models.clear()


def get_loaded_models(detailed: bool = False) -> Dict:
    """ returns the loaded model cache

        Future_Change:
            expose cache related functions to the client
    """
    result = {"models": []}
    for model_cache_key, model in _available_models.items():
        model_name = model_cache_key.split("||")[0]

        if detailed:
            result["models"].append(
                {"model_name": model_name,
                 "model_properties": model.model_properties.model_dump_json(by_alias=True)}
            )
        else:
            result["models"].append({"model_name": model_name})
    return result


def eject_model(model_name: str) -> dict:
    """ ejects a model from the loaded model cache

        Future_Change:
            expose cache related functions to the client
    """
    with _model_op_guard(lock):
        for model_cache_key in list(_available_models.keys()):
            if model_cache_key.startswith(model_name):
                get_available_models()[model_cache_key].unload()
                del _available_models[model_cache_key]
                break
    return {"result": "success", "message": f"Model {model_name} ejected successfully."}