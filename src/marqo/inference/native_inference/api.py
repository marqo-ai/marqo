from fastapi import FastAPI

from marqo import version
from marqo.core.inference.api import *
from marqo.inference.chunk_download_preprocess_content import chunk_download_preprocess_content
from marqo.inference.load_model import load_model
from marqo.inference.encode_content import encode_processed_content, format_results
from torch import Tensor
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
    model = load_model(
        model_name=request.model_config.model_name,
        model_properties=request.model_config.model_properties,
        model_auth=request.model_config.model_auth,
        device=request.device
    )

    # Chunk, download, and preprocess the content
    preprocessed_content_list: List[PreprocessedContent] = chunk_download_preprocess_content(
        content=request.contents,
        modality=request.modality,
        preprocessing_config=request.preprocessing_config,
        preprocessor=model.get_preprocessor(),
    )

    # Encode the processed content
    embeddings: List[Tensor] = encode_processed_content(
        model=model,
        preprocessed_content_list=preprocessed_content_list,
        modality=request.modality,
        normalize=request.model_config.normalize_embeddings
    )

    # Format the results
    formated_result = format_results(preprocessed_content_list, embeddings)
    return formated_result