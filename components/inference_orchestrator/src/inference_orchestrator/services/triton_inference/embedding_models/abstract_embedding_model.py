from abc import ABC, abstractmethod
from typing import List, Optional

from numpy import ndarray

from inference_orchestrator.schemas.api import Modality
from inference_orchestrator.services.triton_inference.model_manager.model_management_client import (
    ModelManagementClient,
)
from inference_orchestrator.services.triton_inference.triton.triton_grpc_client import (
    TritonGRPCClient,
)


class AbstractEmbeddingModel(ABC):
    """This is the abstract base class for all models in Marqo."""

    def __init__(
        self,
        model_properties: dict,
        model_management_client: Optional[ModelManagementClient],
        triton_client: Optional[TritonGRPCClient],
    ):
        """Load the model with the given properties.

        Args:
            model_properties (dict): The properties of the model.
            model_management_client(ModelManagementClient): The client communicating with the
                marqo_model_management_container
            triton_client (TritonGRPCClient): The gRPC client to use for communicating with the Triton Inference Server.
        """

        self.model_properties = model_properties
        self.model_management_client = model_management_client
        self.triton_client = triton_client

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
    def encode(
        self, inputs: List, modality: Modality, normalize: bool
    ) -> List[ndarray]:
        """Encode the input data.

        Args:
            inputs: The input data to be encoded, in the form of a list. The individual elements of the list
                is model specific.
            modality: The modality of the input data.
            normalize: Whether to normalize the embeddings.

        Returns:
            The encoded data. A list of numpy arrays, where each array is the embedding of the corresponding input.
            Thus, each element of the list should be a (Dim, ) array of floats.
            It should be the same length as the input list.
        """
        pass

    @abstractmethod
    def get_preprocessor(self):
        """Get the preprocessor for the model."""
        pass

    @abstractmethod
    def unload(self, remove_files: bool = False):
        """Unload the model from the Triton Inference Server.

        Args:
            remove_files (bool): Whether to remove the model files from disk after unloading.
        """
        pass
