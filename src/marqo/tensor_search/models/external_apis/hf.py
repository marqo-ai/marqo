from typing import Optional

from pydantic import Field

from marqo.tensor_search.models.external_apis.abstract_classes import (
    ObjectLocation, ExternalAuth
)


class HfAuth(ExternalAuth):
    token: str


class HfModelLocation(ObjectLocation):
    repo_id: str = Field(..., description="ID of the repository", alias="repoId")
    filename: Optional[str] = Field(None, description="Name of the file")