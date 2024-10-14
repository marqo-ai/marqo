from abc import ABC, abstractmethod
from typing import Optional, Any
from marqo.tensor_search.models.private_models import ModelAuth

from marqo.core.inference.enums import Modality
import numpy as np


class AbstractEmbeddingModel(ABC):
    """This is the abstract base class for all models in Marqo."""

    def __init__(self, model_properties: Optional[dict] = None, device: Optional[str] = None,
                 model_auth: Optional[ModelAuth] = None):
        """Load the model with the given properties.

        Args:
            model_properties (dict): The properties of the model.
            device (str): The device to load the model on.
            model_auth (dict): The authentication information for the model.
        """
        if device is None:
            raise ValueError("`device` is required for loading CLIP models!")

        if model_properties is None:
            model_properties = dict()

        self.device = device
        self.model_auth = model_auth

    def load(self):
        """Load the model and check if the necessary component are loaded.

        The required components are loaded in the `_load_necessary_components` method.
        The loaded components are checked in the `_check_loaded_components` method.
        """
        self._load_necessary_components()
        self._check_loaded_components()

    @abstractmethod
    def _load_necessary_components(self):
        """Load the necessary components for the model."""
        pass

    @abstractmethod
    def _check_loaded_components(self):
        """Check if the necessary components are loaded.

        Raises:
            A proper exception if the necessary components are not loaded.
        """
        pass

    @abstractmethod
    def _validate_content_type(self, content: Any, modality: Modality):
        """Validate if the provided content type is valid for the specific model and if it matches the modality.

        Raise:
            ValueError: If the content type is not valid.
        """
        pass

    @abstractmethod
    def _encode(self, content: Any, modality: Modality, normalize: bool = True) -> np.ndarray:
        """Encode the given content.

        Args:
            content (Any): The content to encode.
            normalize (bool): Whether to normalize the output or not.
        """
        pass

    @abstractmethod
    def _validate_and_set_modality(self, modality) -> Modality:
        """Validate the modalities for the model.

        We are inferring the modality of the content regardless of the model capabilities. For example, if user provides
        an image url in the search query, we will infer the modality as image even if the model is a text model.

        Returns:
            Modality: The modalities for the model content.

        Raises:
            UnsupportedModalityError: If the model does not support the inferred modality other than text.
        """
        pass

    def encode(self, content: Any, normalize: bool = True, modality: Optional[Modality] = None, **kwargs) -> np.ndarray:
        modality = self._validate_and_set_modality(modality)
        self._validate_content_type(content, modality)
        return self._encode(content, modality, normalize)