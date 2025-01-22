import base64
from typing import Union, List, Optional

from pydantic.main import BaseModel

from marqo.s2_inference.multimodal_model_load import Modality


class VectoriseRequest(BaseModel):
    model_name: str
    modality: Modality = Modality.TEXT
    normalize_embeddings: bool = True
    device: str
    content: Union[str, List[str]]
    # TODO support list[bytes], list[Image] content type
    # TODO support other parameters

