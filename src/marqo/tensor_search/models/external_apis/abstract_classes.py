"""
These are abstract classes that shouldn't be instantiated
"""
from pydantic import BaseModel

class ExternalAuth(BaseModel):
    """Authentication used to download an object
    """
    class Config:
        allow_mutation = False

class ObjectLocation(BaseModel):
    """Reference to an object location (for example a pointer to a model file
    in s3
    """
    class Config:
        allow_mutation = False

