from fastapi import FastAPI

from marqo import version
from marqo.inference.native_inference.inference_pipeline import InferencePipeline
from marqo.inference.type import *

app = FastAPI(
    title="Marqo Native inference API",
    version=version.get_version()
)


# @app.post("/inference")
def inference(request: InferenceRequest) -> InferenceResult:
    """
    The inference endpoint for Marqo Native. This endpoint is used to perform inference on the given request.

    Args:
        request:

    Returns:
        InferenceResult.

    Raises:
        InferencingError: If an error occurs during inference. This is a generic error.
    """
    # Load the model and preprocessor if not already loaded
    return InferencePipeline(request).run_pipeline()