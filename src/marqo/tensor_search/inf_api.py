import os

from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from marqo.logging import get_logger
from marqo.s2_inference import s2_inference
from marqo.tensor_search.main import get_config
from marqo.tensor_search.models.inf_request import VectoriseRequest
from marqo.tensor_search.on_start_script import on_start

logger = get_logger(__name__)


logger.info(f'{os.getpid()}: {__name__} on_start')
on_start(get_config(), 'inference')

inf_app = FastAPI(
    title="Marqo Inference"
)


@inf_app.post("/vectorise")
async def vectorise(request: VectoriseRequest):
    try:
        result = s2_inference.vectorise(
            model_name=request.model_name,
            modality=request.modality,
            normalize_embeddings=request.normalize_embeddings,
            content=request.content,
            device=request.device
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
