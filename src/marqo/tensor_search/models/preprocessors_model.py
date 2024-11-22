from typing import Optional

from torchvision.transforms import Compose

from marqo.base_model import MarqoBaseModel
from marqo.s2_inference.multimodal_model_load import Modality


class Preprocessors(MarqoBaseModel):
    image: Optional[Compose] = None
    text: Optional[Compose] = None
    video: Optional[Compose] = None
    audio: Optional[Compose] = None

    class Config:
        arbitrary_types_allowed = True

    def get_preprocessor(self, modality: Modality):
        if modality == Modality.IMAGE:
            return self.image
        elif modality == Modality.TEXT:
            return self.text
        elif modality == Modality.VIDEO:
            return self.video
        elif modality == Modality.AUDIO:
            return self.audio
        else:
            raise ValueError(f"Unknown modality {modality}")