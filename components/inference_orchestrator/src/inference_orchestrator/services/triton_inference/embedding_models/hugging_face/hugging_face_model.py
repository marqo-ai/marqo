from typing import Callable, List

import numpy as np
from numpy import ndarray
from pydantic import ValidationError
from transformers import AutoTokenizer
from tritonclient.grpc import InferInput, InferRequestedOutput, InferResult

from inference_orchestrator.schemas.api import Modality
from inference_orchestrator.services.errors import (
    InternalServerError,
    InvalidModelPropertiesError,
)
from inference_orchestrator.services.triton_inference.embedding_models.abstract_embedding_model import (
    AbstractEmbeddingModel,
)
from inference_orchestrator.services.triton_inference.embedding_models.abstract_preprocessor import (
    AbstractPreprocessor,
)
from inference_orchestrator.services.triton_inference.embedding_models.hugging_face.hugging_face_model_properties import (
    HuggingFaceModelProperties,
    PoolingMethod,
)
from inference_orchestrator.services.triton_inference.model_manager.model_management_client import (
    ModelManagementClient,
)
from inference_orchestrator.services.triton_inference.triton.triton_grpc_client import (
    TritonGRPCClient,
)

from ..model_download_cache import ModelDownloadCache


class HuggingFacePreprocessor(AbstractPreprocessor):
    """The abstract base class for all Hugging Face preprocessors."""

    def __init__(self):
        super().__init__()

    def preprocess(self, inputs: List[str], modality: Modality) -> List[str]:
        # No preprocessing is needed for Hugging Face models
        return inputs


class HuggingFaceModel(AbstractEmbeddingModel):
    """The concrete class for all sentence transformers models loaded from Hugging Face."""

    def __init__(
        self,
        model_properties: dict,
        model_management_client: ModelManagementClient,
        triton_client: TritonGRPCClient,
    ):
        super().__init__(
            model_properties=model_properties,
            model_management_client=model_management_client,
            triton_client=triton_client,
        )

        self.model_properties: HuggingFaceModelProperties = (
            self._build_model_properties(model_properties)
        )

        self._tokenizer = None
        self._pooling_func = None
        self._preprocessor = HuggingFacePreprocessor()
        self._model = None

    def _build_model_properties(
        self, model_properties: dict
    ) -> HuggingFaceModelProperties:
        """Convert the user input model_properties to HuggingFaceModelProperties."""
        try:
            parsed_properties = HuggingFaceModelProperties(**model_properties)
        except ValidationError as e:
            raise InvalidModelPropertiesError(
                f"Invalid model properties: {model_properties}. Original error {e}"
            ) from e

        return parsed_properties

    def _check_loaded_components(self):
        if self._tokenizer is None:
            raise InternalServerError("Tokenizer is not loaded!")
        if self._pooling_func is None:
            raise InternalServerError("Pooling function is not loaded!")

    def _load_necessary_components(self):
        """Load the necessary components for the hf model.

        Raises:
            InvalidModelPropertiesError: If the model properties are invalid or incomplete.
        """

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_properties.effective_name, cache_dir=ModelDownloadCache.hf_cache_path
        )
        self._pooling_func = self._load_pooling_method()

        self._model = self._load_triton_model()

    def _load_triton_model(self) -> bool:
        """Load the model into Triton Inference Server using the model management client."""
        self.model_management_client.load_model(
            model_properties=self.model_properties.triton_text_encoder_properties.model_dump(
                by_alias=True
            )
        )
        return True

    def _load_pooling_method(self) -> Callable:
        """Load the pooling method for the model."""
        if self.model_properties.pooling_method == PoolingMethod.Mean:
            return self._average_pool_func
        elif self.model_properties.pooling_method == PoolingMethod.CLS:
            return self._cls_pool_func
        else:
            raise InternalServerError(
                f"Invalid pooling method: {self.model_properties.pooling_method}"
            )

    def encode(self, inputs: List[str], modality, normalize=True) -> List[ndarray]:
        if not isinstance(inputs, list) or not isinstance(inputs[0], str):
            raise InternalServerError(
                f"The input data should be a list of strings, but received: {inputs}"
            )

        # Tokenize the input texts
        encoded_input = self._tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.model_properties.tokens,
            return_tensors="np",
        )

        shape = encoded_input["input_ids"].shape

        input_ids = InferInput("input_ids", shape, "INT64")
        input_ids.set_data_from_numpy(encoded_input["input_ids"].astype(np.int64))
        attention_mask = InferInput("attention_mask", shape, "INT64")
        attention_mask.set_data_from_numpy(
            encoded_input["attention_mask"].astype(np.int64)
        )
        token_type_ids = InferInput("token_type_ids", shape, "INT64")
        token_type_ids.set_data_from_numpy(
            encoded_input["token_type_ids"].astype(np.int64)
        )

        outputs = [InferRequestedOutput(name="last_hidden_state")]
        response: InferResult = self.triton_client.encode(
            model_name=self.model_properties.triton_text_encoder_properties.name,
            infer_inputs=[input_ids, attention_mask, token_type_ids],
            infer_outputs=outputs,
        )

        last_hidden_state: ndarray = response.as_numpy("last_hidden_state").copy()
        embeddings = self._pooling_func(
            last_hidden_state, encoded_input["attention_mask"]
        )

        if normalize:
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        return [single_ndarray for single_ndarray in embeddings]

    @staticmethod
    def _average_pool_func(model_output: ndarray, attention_mask):
        """A pooling function that averages the hidden states of the model."""
        attn = attention_mask.astype(np.float32)  # [B, T]
        mask = attn[..., None]  # [B, T, 1]
        summed = (model_output * mask).sum(axis=1)  # [B, H]
        lengths = np.clip(mask.sum(axis=1), 1e-9, None)  # [B, 1]
        emb = summed / lengths  # [B, H]
        return emb

    @staticmethod
    def _cls_pool_func(model_output: np.ndarray, attention_mask):
        """A pooling function that extracts the CLS token from the model output."""
        return model_output[:, 0, :]

    def get_preprocessor(self):
        return self._preprocessor

    def unload(self, remove_files: bool = False):
        for model in [self.model_properties.triton_text_encoder_properties]:
            self.model_management_client.unload_model(
                model.name, remove_files=remove_files
            )
