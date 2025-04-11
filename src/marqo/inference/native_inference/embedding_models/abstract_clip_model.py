from abc import abstractmethod

import numpy as np
import torch
from PIL.Image import Image
from numpy import ndarray

from marqo.inference.media_download_and_preprocess.image_download import (format_and_load_CLIP_images,
                                                                          format_and_load_CLIP_image)
from marqo.inference.native_inference.embedding_models.abstract_embedding_model import AbstractEmbeddingModel
from marqo.inference.native_inference.embedding_models.abstract_preprocessor import AbstractPreprocessor
from marqo.logging import get_logger
from marqo.s2_inference.types import *
from marqo.tensor_search.models.private_models import ModelAuth

logger = get_logger(__name__)


class AbstractCLIPPreprocessor(AbstractPreprocessor):

    def __init__(self, tokenizer, image_preprocessor):
        super().__init__()
        self.tokenizer = tokenizer
        self.image_preprocessor = image_preprocessor

    def preprocess(self, inputs: Union[List[str], List[Image]], modality: Modality):
        if modality == Modality.TEXT:
            return self._tokenize_text(inputs)
        elif modality == Modality.IMAGE:
            return self._preprocess_image(inputs)
        else:
            raise ValueError(f"Unsupported modality: {modality}")

    @abstractmethod
    def _tokenize_text(self, inputs: list[str]) -> List[Tensor]:
        pass

    @abstractmethod
    def _preprocess_image(self, inputs: list[Image]) -> List[Tensor]:
        pass


class AbstractCLIPModel(AbstractEmbeddingModel):
    """Abstract base class for CLIP models.

    Attributes:
        device (str): The device to load the model on, typically 'cpu' or 'cuda'.
        model_properties (dict): A dictionary containing additional properties or configurations
            specific to the model. Defaults to an empty dictionary if not provided.
        model: The actual CLIP model instance, initialized to `None` and to be set by subclasses.
        tokenizer: The tokenizer associated with the model, initialized to `None` and to be set by subclasses.
        image_preprocessor: The image preprocessor associated with the model,
            initialized to `None` and to be set by subclasses.
        preprocessor: The overall preprocessor used by the model that wraps the tokenizer and image preprocessor,
            initialized to `None` and to be set by subclasses.
    """

    def __init__(self, device: Optional[str] = None, model_properties: Optional[dict] = None,
                 model_auth: Optional[ModelAuth] = None):
        """Instantiate the abstract CLIP model.

        Args:
            device (str): The device to load the model on, typically 'cpu' or 'cuda'.
            model_properties (dict): A dictionary containing additional properties or configurations
                specific to the model. Defaults to an empty dictionary if not provided.
            model_auth (ModelAuth): The authentication information for the model. Defaults to `None` if not provided
        """

        super().__init__(model_properties=model_properties, device=device, model_auth=model_auth)

        self.model = None
        self.tokenizer = None
        self.preprocessor = None # The overall preprocessor used by the model that wraps the tokenizer and image preprocessor
        self.image_preprocessor = None # The image preprocessor used by the model

    @abstractmethod
    def encode_text(self, inputs: List, normalize: bool = True) -> List[ndarray]:
        pass

    @abstractmethod
    def encode_image(self, inputs: List, normalize: bool = True) -> List[ndarray]:
        pass

    def encode(self, inputs: List, modality: Modality, normalize=True) -> List[ndarray]:
        if modality == Modality.IMAGE:
            return self.encode_image(inputs, normalize=normalize)
        elif modality == Modality.TEXT:
            return self.encode_text(inputs, normalize=normalize)
        else:
            raise ValueError(f"Unsupported modality: {modality}")

    def _convert_output(self, output: Tensor) -> List[ndarray]:
        if self.device == 'cpu':
            return [single_ndarray for single_ndarray in output.numpy()]
        elif self.device.startswith('cuda'):
            return [single_ndarray for single_ndarray in output.cpu().numpy()]

    @staticmethod
    def normalize(outputs):
        return outputs.norm(dim=-1, keepdim=True)