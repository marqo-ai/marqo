from typing import Literal

from inference_orchestrator.services.triton_inference.embedding_models.base_model_properties import (
    BaseModelProperties,
)


class RandomModelProperties(BaseModelProperties):
    """
    A class to represent the properties of a random model.

    Attributes:
        type: The type of the model. It should be 'random'.
        note: A note about the model. It is optional.
    """

    type: Literal["random"]
