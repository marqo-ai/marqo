from abc import ABC, abstractmethod
from marqo.core.inference.api.modality import Modality
from torch import Tensor


class AbstractPreprocessor(ABC):
    """This is the abstract base class for all preprocessors in Marqo."""

    @abstractmethod
    def preprocess(self, inputs: list, modality: Modality) -> list:
        """Preprocess the input data.

        Args:
            inputs: The input data to be preprocessed.

        Returns:
            The preprocessed data.
        """
        pass