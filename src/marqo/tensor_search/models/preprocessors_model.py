from typing import Optional, Any

from marqo.base_model import MarqoBaseModel
from marqo.core.inference.api.modality import Modality


class Preprocessors(MarqoBaseModel):
    """The type of preprocessors is unknown, so we use Any."""
    image: Optional[Any] = None
    text: Optional[Any] = None
    video: Optional[Any] = None
    audio: Optional[Any] = None

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