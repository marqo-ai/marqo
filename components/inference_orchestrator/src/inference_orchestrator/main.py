from contextlib import asynccontextmanager

import msgpack
import msgpack_numpy
import uvicorn
from fastapi import Body, Depends, FastAPI, HTTPException, Request, Response
from orjson import orjson
from pydantic import ValidationError
from starlette import status
from starlette.responses import JSONResponse

from .api.otel import bootstrap_otel
from .api.telemetry import TelemetryMiddleware
from .config import Config, get_config
from .core.logging import get_logger
from .on_start_script import on_start
from .schemas.api import InferenceRequest
from .services.errors import InternalServerError, ServiceError
from .services.triton_inference.model_manager import model_manager
from .version import get_version

logger = get_logger(__name__)

msgpack_numpy.patch()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Instantiate OpenTelemetry
    otel_shutdown_hook = bootstrap_otel(app, service_name="marqo-inference")

    # Do the on_start tasks
    on_start(get_config())

    yield

    otel_shutdown_hook()
    get_config().triton_client.close()


app = FastAPI(
    title="Marqo Inference",
    lifespan=lifespan,
    version=get_version(),
)

app.add_middleware(TelemetryMiddleware)


def _serialise_error(
    error_response: dict, status_code: int, media_type: str
) -> Response:
    if media_type == "application/msgpack":
        content = msgpack.packb(error_response, use_bin_type=True)
    else:
        content = orjson.dumps(error_response)
    return Response(content=content, status_code=status_code, media_type=media_type)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"Encountered exception: {exc.detail}", exc_info=True)
    media_type = request.headers.get("Accept", "application/json")
    # TODO should we follow the pattern in API?
    error_response = {"detail": exc.detail}
    return _serialise_error(error_response, exc.status_code, media_type)


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Encountered exception: {str(exc)}", exc_info=True)
    media_type = request.headers.get("Accept", "application/json")
    error_response = {"detail": str(exc)}
    return _serialise_error(
        error_response, status.HTTP_500_INTERNAL_SERVER_ERROR, media_type
    )


@app.get("/", summary="Basic information")
def root():
    """
    Used for basic health check
    """
    return {"message": "Welcome to Marqo Inference", "version": app.version}


@app.post("/vectorise")
def vectorise(
    request: Request, raw_body: bytes = Body(...), config: Config = Depends(get_config)
):
    """
    Vectorise a list of contents (str) in a given modality, using the model specified in the request.
    This endpoint expect the reqeust to be encoded in `application/msgpack` media type, and returns the
    result (including errors) in the same media type.
    """
    _check_content_type_msgpack(request)

    # Convert request to InferenceRequest
    try:
        request_data = msgpack.unpackb(raw_body, raw=False)
        inference_request = InferenceRequest.model_validate(request_data)
    except (msgpack.ExtraData, msgpack.UnpackException) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid MessagePack format: {str(e)}",
        ) from e
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e)
        ) from e
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    # Generate embeddings
    try:
        result = config.inference.vectorise(inference_request)
    except InternalServerError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during vectorisation. {e.message}. Please try again later ",
        ) from e
    except ServiceError as e:
        # TODO distinguish recoverable error from unrecoverable error, return different error code
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"An error occurred during vectorisation. {e.message}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during vectorisation. {str(e)}",
        ) from e

    # Converts result to response
    # Serialize the result to a dict and then encode it using MessagePack (with numpy support)
    # TODO if telemetry is set to true, we will need to attach the telemetry data
    result_msgpack = msgpack.packb(result.model_dump(by_alias=True), use_bin_type=True)
    return Response(content=result_msgpack, media_type="application/msgpack")


def _check_content_type_msgpack(request):
    expected_content_type = "application/msgpack"
    content_type = request.headers.get("Content-Type")
    if content_type != expected_content_type:
        logger.warning(f"Unsupported Content-Type: {content_type}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported Content-Type {content_type}. Expected '{expected_content_type}'.",
        )


@app.get("/healthz", include_in_schema=False)
def liveness_check() -> JSONResponse:
    """
    This liveness check endpoint does a quick status check, and error out if any component encounters unrecoverable
    issues. This only does a check on the cuda devices right now.
    Docker schedulers could leverage this endpoint to decide whether to restart the Marqo container.

    Returns:
        200 - if all checks pass
        500 - if any check fails
    """
    return JSONResponse(content={"status": "ok"}, status_code=200)


@app.get("/models")
def get_loaded_models(detailed: bool = False):
    return model_manager.get_loaded_models(detailed=detailed)


@app.delete("/models")
def eject_model(model_name: str):
    return model_manager.eject_model(model_name)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8884)
