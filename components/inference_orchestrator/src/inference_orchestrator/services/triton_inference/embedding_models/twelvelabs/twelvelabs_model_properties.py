from typing import Literal, Optional

from pydantic import Field

from inference_orchestrator.services.triton_inference.embedding_models.base_model_properties import (
    BaseModelProperties,
)


class TwelveLabsModelProperties(BaseModelProperties):
    """
    Properties for a TwelveLabs Marengo embedding model.

    Marengo is served by the TwelveLabs API rather than by Triton, so no
    ``tritonImageEncoderProperties``/``tritonTextEncoderProperties`` are
    required. The model produces 512 dimensional multimodal embeddings for
    text, image and video content.

    Attributes:
        type: The type of the model. It must be ``twelvelabs``.
        api_model_name: The model name passed to the TwelveLabs API
            (e.g. ``marengo3.0``). Defaults to ``marengo3.0``.
    """

    type: Literal["twelvelabs"]
    api_model_name: str = Field(default="marengo3.0", alias="apiModelName")
    note: Optional[str] = None
