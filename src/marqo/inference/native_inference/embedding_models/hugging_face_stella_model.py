from typing import Optional

from marqo.inference.native_inference.embedding_models.hugging_face_model import HuggingFaceModel
from marqo.inference.native_inference.embedding_models.hugging_face_model_properties import HuggingFaceModelFlags, \
    HuggingFaceTokenizerFlags
from marqo.tensor_search.models.private_models import ModelAuth


class HuggingFaceStellaModel(HuggingFaceModel):
    """The concrete class for Stella models loaded from Hugging Face."""

    def __init__(self, model_properties: dict, device: str, model_auth: Optional[ModelAuth] = None):
        super().__init__(
            model_properties, device, model_auth,
            model_flags=HuggingFaceModelFlags(
                trust_remote_code=True,
                use_memory_efficient_attention=False,
                unpad_inputs=False
            ),
            tokenizer_flags=HuggingFaceTokenizerFlags(
                trust_remote_code=True
            )
        )
