from pydantic.dataclasses import dataclass
from typing import Optional
from marqo.tensor_search.models.external_apis.abstract_classes import (
    ObjectLocation, ExternalAuth
)
from pydantic import BaseModel, Field, validator
from marqo.api.exceptions import InvalidArgError


class HfAuth(ExternalAuth):
    token: str


class HfModelLocation(ObjectLocation):
    repo_id: str = Field(..., description="ID of the repository")
    filename: Optional[str] = Field(None, description="Name of the file")