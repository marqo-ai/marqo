from fastapi import APIRouter, Depends, Query

from ..config import Config, get_config
from ..schemas.load_model_request import LoadModelRequest

router = APIRouter(prefix="/v1", tags=["v1"])


@router.post("/models/load")
def load_model(payload: LoadModelRequest, cfg: Config = Depends(get_config)):
    """
    Load a model into the Triton Inference Server.
    :param payload: the model properties to load
    :return: 200 OK if the model was loaded successfully
    """
    cfg.model_manager.load_model(payload.triton_model_properties)


@router.post("/models/{model_name}/unload")
def unload_model(
        model_name: str, remove_files: bool = Query(False, alias="remove-files"),
        cfg: Config = Depends(get_config)
):
    """
    Unload a model from the Triton Inference Server.
    :param model_name: the name of the model to unload
    :param remove_files: Whether to remove the model files from disk after unloading
    :return: 200 OK if the model was unloaded successfully or if the model was not found
    """
    cfg.model_manager.unload_model(model_name, remove_files=remove_files)

