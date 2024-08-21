from abc import ABC, abstractmethod


class AbstractModel(ABC):
    """This is the abstract base class for all models in Marqo."""

    @abstractmethod
    def load(self):
        """Load the model."""
        pass

    @abstractmethod
    def encode(self):
        pass