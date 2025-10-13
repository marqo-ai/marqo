"""
These are abstract classes that shouldn't be instantiated
"""
from marqo.base_model import ImmutableBaseModel

class ExternalAuth(ImmutableBaseModel):
    """Authentication used to download an object
    """

class ObjectLocation(ImmutableBaseModel):
    """Reference to an object location (for example a pointer to a model file
    in s3
    """
