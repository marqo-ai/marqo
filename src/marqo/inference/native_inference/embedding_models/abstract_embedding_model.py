from abc import ABC, abstractmethod
from typing import Optional

from marqo.core.inference.api.inference import ModelAuth
from marqo.inference.native_inference.embedding_models.abstract_preprocessor import AbstractPreprocessor


class AbstractEmbeddingModel(ABC):
    """This is the abstract base class for all models in Marqo."""

    def __init__(self, model_properties: dict, device: str, model_auth: Optional[ModelAuth] = None):
        """Load the model with the given properties.

        Args:
            model_properties (dict): The properties of the model.
            device (str): The device to load the model on.
            model_auth (dict): The authentication information for the model.
        """

        self.model_properties = model_properties
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
    def encode(self, inputs, modality, normalize):
        """Encode the input data."""
        # downloading and preprocess inside the encode method
        pass

    @abstractmethod
    def get_preprocessor(self)-> AbstractPreprocessor:
        """Get the preprocessor for the model."""
        pass