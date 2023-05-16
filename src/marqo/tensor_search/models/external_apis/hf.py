from pydantic.dataclasses import dataclass
from typing import Optional
from marqo.tensor_search.models.external_apis.abstract_classes import (
    ObjectLocation, ExternalAuth
)
from pydantic import BaseModel, Field, validator
from marqo.errors import InvalidArgError


class HfAuth(ExternalAuth):
    token: str


class HfModelLocation(ObjectLocation):
    repo_id: str = Field(..., description="ID of the repository")
    filename: Optional[str] = Field(None, description="Name of the file")

    @validator('repo_id')
    def validate_repo_id(cls, value):
        if not isinstance(value, str):
            raise InvalidArgError("To load a custom model from huggingface, `repo_id` must be provided as a valid string.")
        return value