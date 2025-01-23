import os
from typing import List

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from marqo.logging import get_logger
from marqo.s2_inference import s2_inference
from marqo.tensor_search.main import get_config
from marqo.tensor_search.models.inf_request import VectoriseRequest
from marqo.tensor_search.on_start_script import on_start, StartMode

logger = get_logger(__name__)


logger.info(f'{os.getpid()}: {__name__} on_start')
on_start(get_config(), StartMode.INFERENCE)

inf_app = FastAPI(
    title="Marqo Inference"
)


@inf_app.post("/vectorise")
async def vectorise(request: VectoriseRequest):
    try:
        result: List[List[float]] = s2_inference.vectorise(
            model_name=request.model_name,
            model_properties=request.model_properties,
            model_auth=request.model_auth,
            modality=request.modality,
            normalize_embeddings=request.normalize_embeddings,
            content=request.content,
            device=request.device,
            enable_cache=request.enable_cache,
            media_download_headers=request.media_download_headers,
        )

        return ORJSONResponse(result)

    except Exception as e:
        logger.error(e)
        return {"error": f"Error during vectorization: {str(e)}"}

        # TODO pass errors back to client:
        # s2_inference_errors.UnknownModelError,
        # s2_inference_errors.InvalidModelPropertiesError,
        # s2_inference_errors.ModelLoadError,
        # s2_inference.ModelDownloadError,
        # s2_inference_errors.S2InferenceError
