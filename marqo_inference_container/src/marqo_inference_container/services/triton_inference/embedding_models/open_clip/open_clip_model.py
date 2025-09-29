from typing import List, Optional, Dict, Callable, Tuple

import numpy as np
import open_clip
import torch
from PIL.Image import Image
from numpy import ndarray
from open_clip.transform import image_transform_v2
from pydantic.v1 import ValidationError
from torch import Tensor
from torchvision.transforms import Compose
from tritonclient.grpc import InferInput, InferRequestedOutput, InferResult

from marqo_inference_container import marqo_docs
from marqo_inference_container.core.logging import get_logger
from marqo_inference_container.errors.common_errors import InternalError
from marqo_inference_container.errors.common_errors import InvalidModelPropertiesError
from marqo_inference_container.services.triton_inference.embedding_models.abstract_clip_model import AbstractCLIPModel
from marqo_inference_container.services.triton_inference.embedding_models.abstract_clip_model import \
    AbstractCLIPPreprocessor
from marqo_inference_container.services.triton_inference.embedding_models.model_download_cache import ModelDownloadCache
from marqo_inference_container.services.triton_inference.embedding_models.open_clip.hf_tokenizer import HFTokenizer
from marqo_inference_container.services.triton_inference.embedding_models.open_clip.open_clip_model_properties import \
    OpenCLIPModelProperties
from marqo_inference_container.services.triton_inference.model_manager.model_manager import TritonModelManager
from marqo_inference_container.services.triton_inference.triton.triton_grpc_client import TritonGRPCClient

logger = get_logger(__name__)

HF_HUB_PREFIX = "hf-hub:"
MARQO_OPEN_CLIP_REGISTRY_PREFIX = "open_clip/"


class OpenCLIPPreprocessor(AbstractCLIPPreprocessor):

    def __init__(self, tokenizer, image_preprocessor, device: str):
        super().__init__(tokenizer, image_preprocessor)
        self.device = device

    def _tokenize_text(self, inputs: list[str]) -> list[str]:
        """
        Preprocess the text using the tokenizer.
        Args:
            inputs: A list of strings to preprocess.

        Returns:
            A list of strings. We leave the model.encode_text to handle the tokenization.
        """
        return inputs

    def _preprocess_image(self, inputs: list[Image]) -> List[Tensor]:
        """
        Preprocess the images using the image preprocessor.
        Args:
            inputs: A list of images to preprocess.

        Returns:
            A list of preprocessed images in the form of tensors.
            Each tensor has the shape (N, 3, H, W) where N is the batch_size,
             H and W are the height and width of the image.
        """
        # Need unsqueeze(0) to add the batch dimension
        return [self.image_preprocessor(image).unsqueeze(0).to(self.device) for image in inputs]


class OpenCLIPModel(AbstractCLIPModel):
    def __init__(
            self,
            triton_client,
            model_properties: Optional[Dict] = None,
            model_auth=None,
            model_manager=None,
    ) -> None:

        super().__init__(device="cpu", model_properties=model_properties, model_auth=model_auth)

        self.model_properties = self._build_model_properties(model_properties)
        self.triton_client: TritonGRPCClient = triton_client
        self.model_manager: TritonModelManager = model_manager

        self.image_preprocessor_config = None

    def _build_model_properties(self, model_properties: dict) -> OpenCLIPModelProperties:
        """Convert the user input model_properties to OpenCLIPModelProperties."""
        try:
            return OpenCLIPModelProperties(**model_properties)
        except ValidationError as e:
            raise InvalidModelPropertiesError(f"Invalid model properties: {model_properties}. Original error: {e}") \
                from e

    def _load_necessary_components(self) -> None:
        """Load the open_clip model and tokenizer."""
        if self.model_properties.name.startswith(HF_HUB_PREFIX):
            _, self.image_preprocessor = self._load_model_and_image_preprocessor_from_hf_repo()
            self.tokenizer = self._load_tokenizer_from_hf_repo()
        elif self.model_properties.name.startswith(MARQO_OPEN_CLIP_REGISTRY_PREFIX):
            _, self.image_preprocessor = self._load_model_and_image_preprocessor_from_open_clip_repo()
            self.tokenizer = self._load_tokenizer_from_open_clip_repo()
        else:
            raise InvalidModelPropertiesError(
                f"Marqo cannot load the provided open_clip model. "
                f"Check {marqo_docs.bring_your_own_model()} "
                f"for more details on the supported methods to open_clip model "
            )

        self.model = self._load_triton_model()
        self.preprocessor = OpenCLIPPreprocessor(self.tokenizer, self.image_preprocessor, device=self.device)

    def _load_triton_model(self) -> None:
        self.model_manager.load_model(
            self.model_properties.triton_image_encoder.model_dump(by_alias=True)
        )

        self.model_manager.load_model(
            self.model_properties.triton_text_encoder.model_dump(by_alias=True)
        )

        return True

    def get_preprocessor(self) -> OpenCLIPPreprocessor:
        return self.preprocessor

    def _check_loaded_components(self):
        """Check if the open_clip model, tokenizer, and image preprocessor are loaded.

        Raises:
            RuntimeError: If the open_clip model, tokenizer, or image preprocessor is not loaded.
        """
        if self.model is None:
            raise RuntimeError("The open_clip model is not loaded. Please load the model before inference.")
        if self.tokenizer is None:
            raise RuntimeError("The open_clip tokenizer is not loaded. Please load the tokenizer before inference.")
        if self.image_preprocessor is None:
            raise RuntimeError("The open_clip image preprocessor is not loaded. "
                               "Please load the image preprocessor before inference.")

    def _load_image_preprocessor(self) -> Callable:
        return image_transform_v2(self.image_preprocessor_config)

    def _load_model_and_image_preprocessor_from_hf_repo(self) -> Tuple[torch.nn.Module, Compose]:
        """Load the model and image preprocessor from a hf_repo.

        The hf_repo should be provided in the model properties, and it is a string starting with `hf-hub:`.
        """
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=self.model_properties.name,
            device=self.device,
            cache_dir=ModelDownloadCache.open_clip_cache_path,
        )
        return model, preprocess

    def _load_model_and_image_preprocessor_from_open_clip_repo(self) -> Tuple[torch.nn.Module, Compose]:
        """Load the model and image preprocessor from the marqo model registry.

        The model name should be provided in the model properties, and it is a string starting with `open_clip/`.
        """
        architecture = self.model_properties.name.split("/", 3)[1]
        pretrained = self.model_properties.name.split("/", 3)[2]

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=architecture,
            pretrained=pretrained,
            device=self.device,
            cache_dir=ModelDownloadCache.open_clip_cache_path
        )
        return model, preprocess

    def _load_tokenizer_from_checkpoint(self) -> Callable:
        if not self.model_properties.tokenizer:
            if self.model_properties.name.startswith(HF_HUB_PREFIX):
                return open_clip.get_tokenizer(self.model_properties.name)
            else:
                # Replace '/'with '-' to support old clip model name style
                return open_clip.get_tokenizer(self.model_properties.name.replace("/", "-"))
        else:
            logger.info(f"Custom HFTokenizer is provided. Loading...")
            return HFTokenizer(self.model_properties.tokenizer)

    def _load_tokenizer_from_hf_repo(self) -> Callable:
        return open_clip.get_tokenizer(self.model_properties.name)

    def _load_tokenizer_from_open_clip_repo(self) -> Callable:
        return open_clip.get_tokenizer(self.model_properties.name.split("/", 3)[1])

    def encode_image(self, images: List[Tensor], normalize=True) -> List[ndarray]:

        images = torch.cat(images, dim=0).numpy().astype(self.model_properties.image_input_numpy_type)
        inputs = np.ascontiguousarray(images, dtype=self.model_properties.image_input_numpy_type)

        input_tensor = InferInput(
            name=self.model_properties.triton_image_encoder.input[0].name,
            shape=list(inputs.shape),
            datatype=self.model_properties.image_input_triton_type
        )
        input_tensor.set_data_from_numpy(inputs)

        output_tensor = InferRequestedOutput(
            name=self.model_properties.triton_image_encoder.output[0].name
        )

        response: InferResult = self.triton_client.encode(
            model_name=self.model_properties.triton_image_encoder.name,
            infer_inputs=[input_tensor],
            infer_outputs=[output_tensor]
        )

        # Do a copy to ensure it is writable
        embeddings = response.as_numpy(self.model_properties.triton_image_encoder.output[0].name).copy()

        if normalize:
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        if embeddings.shape != (len(images), self.model_properties.dimensions):
            raise InternalError(
                f"The shape of the text embeddings {embeddings.shape} does not match the expected shape "
                f"({len(images)}, {self.model_properties.dimensions})"
            )
        return [embeddings[i] for i in range(embeddings.shape[0])]

    def encode_text(self, text: list[str], normalize=True) -> List[ndarray]:
        tokenized_text = self.tokenizer(text).reshape(len(text), -1).numpy()
        inputs = np.ascontiguousarray(tokenized_text, dtype=self.model_properties.text_input_numpy_type)

        input_tensor = InferInput(
            name=self.model_properties.triton_text_encoder.input[0].name,
            shape=list(inputs.shape),
            datatype=self.model_properties.text_input_triton_type
        )
        input_tensor.set_data_from_numpy(inputs)

        output_tensor = InferRequestedOutput(
            name=self.model_properties.triton_text_encoder.output[0].name
        )

        response: InferResult = self.triton_client.encode(
            model_name=self.model_properties.triton_text_encoder.name,
            infer_inputs=[input_tensor],
            infer_outputs=[output_tensor]
        )

        # Do a copy to ensure it is writable
        embeddings = response.as_numpy(self.model_properties.triton_text_encoder.output[0].name).copy()

        if normalize:
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        if embeddings.shape != (len(text), self.model_properties.dimensions):
            raise InternalError(
                f"The shape of the text embeddings {embeddings.shape} does not match the expected shape "
                f"({len(text)}, {self.model_properties.dimensions})"
            )
        return [embeddings[i] for i in range(embeddings.shape[0])]
