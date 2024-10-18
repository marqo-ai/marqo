from typing import Optional

<<<<<<< Updated upstream
from marqo.core.inference.inference_models.hugging_face_model import HuggingFaceModel
from marqo.core.inference.inference_models.hugging_face_model_properties import HuggingFaceModelFlags, \
    HuggingFaceTokenizerFlags
from marqo.tensor_search.models.private_models import ModelAuth
=======
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import ValidationError
from sympy.codegen.fnodes import dimension
from transformers import (AutoModel, AutoTokenizer)

from marqo import marqo_docs
from marqo.core.inference.inference_models.hugging_face_model import HuggingFaceModel
from marqo.core.inference.model_download import download_model
from marqo.core.inference.inference_models.abstract_embedding_model import AbstractEmbeddingModel
from marqo.core.inference.inference_models.hugging_face_model_properties import HuggingFaceModelProperties, \
    PoolingMethod
from marqo.s2_inference.configs import ModelCache
from marqo.s2_inference.errors import InvalidModelPropertiesError
from marqo.s2_inference.types import Union, FloatTensor, List
>>>>>>> Stashed changes


class HuggingFaceStellaModel(HuggingFaceModel):
    """The concrete class for Stella models loaded from Hugging Face."""

<<<<<<< Updated upstream
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
=======

def _cls_pool_func(model_output, attention_mask=None):
    return model_output[0][:, 0]


class HuggingFaceStellaModel(HuggingFaceModel):
    """The concrete class for all sentence transformers models loaded from Hugging Face."""

    def __init__(self, model_properties: dict, device: str, model_auth: dict):
        super().__init__(model_properties, device, model_auth)

        self.model_properties = self._build_model_properties(model_properties)

        self.model = None
        self.tokenizer = None
        self.vector_linear = None
        self.pooling_func = None

    def _build_model_properties(self, model_properties: dict) -> HuggingFaceModelProperties:
        try:
            return HuggingFaceModelProperties(**model_properties)
        except ValidationError as e:
            raise InvalidModelPropertiesError(f"Invalid model properties for the 'hf' model. Original error {e}") \
                from e

    def _check_loaded_components(self):
        if self.model is None:
            raise RuntimeError("Model is not loaded!")
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded!")
        if self.pooling_func is None:
            raise RuntimeError("Pooling function is not loaded!")

    def _load_necessary_components(self):
        if self.model_properties.name:
            self.model, self.tokenizer = self._load_from_hugging_face_repo()
        elif self.model_properties.url:
            self.model, self.tokenizer, self.vector_linear = self._load_from_zip_file()
        elif self.model_properties.model_location:
            if self.model_properties.model_location.s3:
                self.model, self.tokenizer = self._load_from_zip_file()
            elif self.model_properties.model_location.hf:
                if self.model_properties.model_location.hf.filename:
                    self.model, self.tokenizer = self._load_from_zip_file()
                else:
                    self.model, self.tokenizer = self._load_from_private_hugging_face_repo()
            else:
                raise InvalidModelPropertiesError(
                    f"Invalid model properties for the 'hf' model. "
                    f"You do not have the necessary information to load the model. "
                    f"Check {marqo_docs.bring_your_own_model()} for more information."
                )
        else:
            raise InvalidModelPropertiesError(
                f"Invalid model properties for the 'hf' model. "
                f"You do not have the necessary information to load the model. "
                f"Check {marqo_docs.bring_your_own_model()} for more information."
>>>>>>> Stashed changes
            )
        )
