from pydantic import BaseModel
from torchvision.transforms import Compose
from typing import Optional


class Preprocessors(BaseModel):
    image: Optional[Compose] = None
    text: Optional[Compose] = None
    video: Optional[Compose] = None
    audio: Optional[Compose] = None

    class Config:
        arbitrary_types_allowed = True