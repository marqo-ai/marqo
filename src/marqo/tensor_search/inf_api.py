import io
import os
import time
from typing import List
import torch

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import ORJSONResponse
from orjson import orjson

from marqo.logging import get_logger
from marqo.s2_inference import s2_inference
from marqo.s2_inference.multimodal_model_load import Modality
from marqo.tensor_search.main import get_config
from marqo.tensor_search.models.inf_request import VectoriseRequest, VectoriseResponse
from marqo.tensor_search.on_start_script import on_start, StartMode

logger = get_logger(__name__)


logger.info(f'{os.getpid()}: {__name__} on_start')
on_start(get_config(), StartMode.INFERENCE)

inf_app = FastAPI(
    title="Marqo Inference"
)


def tensor_from_json(tensor_dict):
    # Extract data, dtype, and shape
    data = tensor_dict['data']
    dtype_str = tensor_dict['dtype']
    shape = tensor_dict['shape']
    # Convert dtype string back to torch dtype
    dtype = getattr(torch, dtype_str.replace('torch.', ''))
    # Reconstruct the tensor
    tensor = torch.tensor(data, dtype=dtype)
    # Ensure the tensor has the correct shape
    tensor = tensor.reshape(shape)

    return tensor


async def tensor_from_file(tensor_file):
    tensor_buffer = io.BytesIO(await tensor_file.read())
    return torch.load(tensor_buffer)


@inf_app.post("/vectorise-binary")
async def vectorise_binary(
        metadata: str = Form(...),
        tensor_file: UploadFile = File(...)
):
    try:
        request = VectoriseRequest(**orjson.loads(metadata))
        return vectorise_internal(request, await tensor_from_file(tensor_file))
    except Exception as e:
        logger.error(e)
        raise e


@inf_app.post("/vectorise")
def vectorise(request: VectoriseRequest):
    try:
        content = tensor_from_json(request.content) if request.preprocessed else request.content
        return vectorise_internal(request, content)
    except Exception as e:
        logger.error(e)
        raise e

        # TODO pass errors back to client:
        # s2_inference_errors.UnknownModelError,
        # s2_inference_errors.InvalidModelPropertiesError,
        # s2_inference_errors.ModelLoadError,
        # s2_inference.ModelDownloadError,
        # s2_inference_errors.S2InferenceError


def vectorise_internal(request, content):
    infer = request.modality == Modality.IMAGE
    start_time = time.perf_counter()
    result: List[List[float]] = s2_inference.vectorise(
        model_name=request.model_name,
        model_properties=request.model_properties,
        model_auth=request.model_auth,
        modality=request.modality,
        normalize_embeddings=request.normalize_embeddings,
        content=content,
        # TODO unless device is set, the best available device should be used
        device=request.device,
        enable_cache=request.enable_cache,
        media_download_headers=request.media_download_headers,
        infer=infer
    )
    vectorise_time = (time.perf_counter() - start_time) * 1000
    return ORJSONResponse(VectoriseResponse(embeddings=result, vectorise_time=vectorise_time).dict())
