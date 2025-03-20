import pydantic
from orjson import orjson
from starlette import status

from marqo import version, logging
from marqo.inference.native_inference.remote.server.inference_config import Config
from marqo.inference.native_inference.remote.server.on_start_script import on_start
from marqo.tensor_search.telemetry import TelemetryMiddleware
from fastapi import FastAPI, Request, Response, Depends, HTTPException, Body
from marqo.core.inference.api import InferenceRequest, InferenceError
import uvicorn
import msgpack
import msgpack_numpy

msgpack_numpy.patch()


logger = logging.get_logger(__name__)

_config = Config()
if __name__ in ["__main__", "api"]:
    on_start(_config)
app = FastAPI(
    name='Marqo Inference',
    version=version.get_version(),
)
app.add_middleware(TelemetryMiddleware)


def get_config():
    return _config


def _serialise_error(error_response: dict, status_code: int, media_type: str) -> Response:
    if media_type == 'application/msgpack':
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
async def http_exception_handler(request: Request, exc: Exception):
    logger.error(f"Encountered exception: {str(exc)}", exc_info=True)
    media_type = request.headers.get("Accept", "application/json")
    error_response = {"detail": str(exc)}
    return _serialise_error(error_response, status.HTTP_500_INTERNAL_SERVER_ERROR, media_type)


@app.on_event("shutdown")
def shutdown_event():
    """clean up on shutdown."""
    pass


@app.get("/", summary="Basic information")
def root():
    """
    Used for basic health check
    """
    return {"message": "Welcome to Marqo Inference",
            "version": app.version}


@app.post("/vectorise")
def vectorise(request: Request, raw_body: bytes = Body(...), config: Config = Depends(get_config)):
    """
    Vectorise a list of contents (str) in a given modality, using the model specified in the request.
    This endpoint expect the reqeust to be encoded in `application/msgpack` media type, and returns the
    result (including errors) in the same media type.
    """
    _check_content_type_msgpack(request)

    # Convert request to InferenceRequest
    try:
        request_data = msgpack.unpackb(raw_body, raw=False)
        inference_request = InferenceRequest.parse_obj(request_data)
    except (msgpack.ExtraData, msgpack.UnpackException) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid MessagePack format: {str(e)}"
        ) from e
    except pydantic.ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.errors()
        ) from e

    # Generate embeddings
    try:
        result = config.local_inference.vectorise(inference_request)
    except InferenceError as e:
        # TODO distinguish recoverable error from unrecoverable error, return different error code
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"An error occurred during vectorisation. {e.message}"
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during vectorisation. {str(e)}"
        ) from e

    # Converts result to response
    # Serialize the result to a dict and then encode it using MessagePack (with numpy support)
    # TODO if telemetry is set to true, we will need to attach the telemetry data
    result_msgpack = msgpack.packb(result.dict(), use_bin_type=True)
    return Response(content=result_msgpack, media_type="application/msgpack")


def _check_content_type_msgpack(request):
    expected_content_type = 'application/msgpack'
    content_type = request.headers.get('Content-Type')
    if content_type != expected_content_type:
        logger.warning(f"Unsupported Content-Type: {content_type}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported Content-Type {content_type}. Expected '{expected_content_type}'."
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8881)
