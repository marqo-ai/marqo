from typing import List, Callable, Tuple

import numpy as np
import open_clip
import torch
from PIL.Image import Image
from numpy import ndarray
from open_clip.transform import image_transform_v2
from pydantic import ValidationError
from torch import Tensor
from torchvision.transforms import Compose
from tritonclient.grpc import InferInput, InferRequestedOutput, InferResult

from inference_orchestrator import marqo_docs
from inference_orchestrator.core.logging import get_logger
from inference_orchestrator.schemas.api import Modality
from inference_orchestrator.services.errors import InternalServerError, InvalidModelPropertiesError
from inference_orchestrator.services.triton_inference.embedding_models.abstract_embedding_model import \
    AbstractEmbeddingModel
from inference_orchestrator.services.triton_inference.embedding_models.abstract_preprocessor import \
    AbstractPreprocessor
from inference_orchestrator.services.triton_inference.embedding_models.model_download_cache import ModelDownloadCache
from inference_orchestrator.services.triton_inference.embedding_models.open_clip.hf_tokenizer import HFTokenizer
from inference_orchestrator.services.triton_inference.embedding_models.open_clip.open_clip_model_properties import \
    OpenCLIPModelProperties
from inference_orchestrator.services.triton_inference.model_manager.model_management_client import ModelManagementClient
from inference_orchestrator.services.triton_inference.triton.triton_grpc_client import TritonGRPCClient

logger = get_logger(__name__)

HF_HUB_PREFIX = "hf-hub:"
MARQO_OPEN_CLIP_REGISTRY_PREFIX = "open_clip/"


class OpenCLIPPreprocessor(AbstractPreprocessor):

    def __init__(self, tokenizer, image_preprocessor):
        super().__init__()
        self.tokenizer = tokenizer
        self.image_preprocessor = image_preprocessor

    def preprocess(self, inputs: list, modality: Modality) -> list:
        """
        Preprocess the inputs based on the modality.
        Args:
            inputs: A list of inputs to preprocess. The individual elements of the list
                is model specific.
            modality: The modality of the input data. It can be either 'text' or 'image'.

        Returns:
            A list of preprocessed inputs. The individual elements of the list
                is model specific.
        """
        if modality == Modality.TEXT:
            return self._tokenize_text(inputs)
        elif modality == Modality.IMAGE:
            return self._preprocess_image(inputs)
        else:
            raise InternalServerError(
                f"Unsupported modality: {modality}. Supported modalities are '{Modality.TEXT.value}'and "
                f"'{Modality.IMAGE.value}', but received '{modality.value}' "
            )

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
        return [self.image_preprocessor(image).unsqueeze(0) for image in inputs]


class OpenCLIPModel(AbstractEmbeddingModel):
    def __init__(
            self,
            model_properties: dict,
            model_management_client: ModelManagementClient,
            triton_client: TritonGRPCClient,

    ) -> None:
        super().__init__(model_properties=model_properties, model_management_client=model_management_client, triton_client=triton_client)

        self.model_properties = self._build_model_properties(self.model_properties)
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
        self.preprocessor = OpenCLIPPreprocessor(self.tokenizer, self.image_preprocessor)

    def _load_triton_model(self) -> bool:
        self.model_management_client.load_model(
            self.model_properties.triton_image_encoder.model_dump(by_alias=True)
        )

        self.model_management_client.load_model(
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
            device="cpu",
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
            device="cpu",
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

    def encode(self, inputs: List, modality: Modality, normalize: bool) -> List[ndarray]:
        if modality == Modality.TEXT:
            return self.encode_text(inputs, normalize=normalize)
        elif modality == Modality.IMAGE:
            return self.encode_image(inputs, normalize=normalize)
        else:
            raise InternalServerError(
                f"Unsupported modality: {modality}. Supported modalities are '{Modality.TEXT.value}'and "
                f"'{Modality.IMAGE.value}', but received '{modality.value}' "
            )

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
            raise InternalServerError(
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
            raise InternalServerError(
                f"The shape of the text embeddings {embeddings.shape} does not match the expected shape "
                f"({len(text)}, {self.model_properties.dimensions})"
            )
        return [embeddings[i] for i in range(embeddings.shape[0])]

    def unload(self, remove_files: bool = False):
        for model in [self.model_properties.triton_image_encoder, self.model_properties.triton_text_encoder]:
            self.model_management_client.unload_model(model.name, remove_files=remove_files)
