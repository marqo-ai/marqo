from typing import Literal

from inference_orchestrator.services.triton_inference.embedding_models.base_model_properties import (
    BaseModelProperties,
)


class RandomModelProperties(BaseModelProperties):
    """
    A class to represent the properties of a random model.

    Attributes:
        name: The name of the model. It can be the name of the model for loading information. e.g., the
            architecture name of the model, the name of the model in the Hugging Face model hub, etc. It might be
            the same as the model tag but this is not necessary.
        type: The type of the model. It should be 'random'.
        note: A note about the model. It is optional.
    """

    name: str
    type: Literal["random"]
