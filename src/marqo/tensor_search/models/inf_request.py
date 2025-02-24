import base64
from typing import Union, List, Optional

from pydantic.main import BaseModel

from marqo.s2_inference.multimodal_model_load import Modality
from marqo.tensor_search.models.private_models import ModelAuth


class VectoriseRequest(BaseModel):
    model_name: str
    model_properties: dict
    model_auth: ModelAuth = None
    modality: Modality = Modality.TEXT
    normalize_embeddings: bool = True
    device: str
    content: Optional[Union[str, List[str]]]
    enable_cache: bool = False,
    media_download_headers: Optional[dict] = None

    preprocessed: bool = False

    # TODO support other fields in **kwargs

    # TODO support other content types: List[Image], List[bytes], etc.


class VectoriseResponse(BaseModel):
    embeddings: List[List[float]]
    vectorise_time: float
