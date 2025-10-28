from fastapi import APIRouter, Depends, Query, Request
from fastapi.routing import APIRoute
from model_management.core.logging import get_logger

from ..config import Config, get_config
from ..schemas.api_models import (
    LoadModelRequest,
    LoadModelResponse,
    UnloadModelResponse,
)

logger = get_logger(__name__)


class MarqoCustomRoute(APIRoute):
    """This is a custom route that logs the error and raises it.

    The log will include the stack trace of the error for debugging purposes.
    The raised error will be handled by the exception handlers. We DO NOT handle the error here.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_route_handler(self):
        original_route_handler = super().get_route_handler()

        async def marqo_custom_route_handler(request: Request):
            try:
                return await original_route_handler(request)
            except Exception as exc:
                logger.error(str(exc), exc_info=True)
                raise exc

        return marqo_custom_route_handler


router = APIRouter(prefix="/v1", tags=["v1"], route_class=MarqoCustomRoute)


@router.get("/healthz", include_in_schema=False)
def liveness_check():
    """
    The liveness check endpoint for the model management service.
    :return: 200 OK if the service is alive
    """
    return {"status": "ok"}


@router.post("/models/load", response_model=LoadModelResponse)
def load_model(payload: LoadModelRequest, cfg: Config = Depends(get_config)):
    """
    Load a model into the Triton Inference Server.
    :param payload: the model properties to load
    :return: 200 OK if the model was loaded successfully
    """
    cfg.model_manager.load_model(payload.triton_model_properties)
    return LoadModelResponse(
        message=f"Model '{payload.triton_model_properties.name}' loaded successfully."
    )


@router.post("/models/{model_name}/unload", response_model=UnloadModelResponse)
def unload_model(
    model_name: str,
    remove_files: bool = Query(False, alias="remove-files"),
    cfg: Config = Depends(get_config),
):
    """
    Unload a model from the Triton Inference Server.
    :param model_name: the name of the model to unload
    :param remove_files: Whether to remove the model files from disk after unloading
    :return: 200 OK if the model was unloaded successfully or if the model was not found
    """
    cfg.model_manager.unload_model(model_name, remove_files=remove_files)
    return UnloadModelResponse(message=f"Model '{model_name}' unloaded successfully.")
