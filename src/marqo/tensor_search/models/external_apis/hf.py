from pydantic.dataclasses import dataclass
from marqo.tensor_search.models.external_apis.abstract_classes import (
    ObjectLocation, ExternalAuth
)


@dataclass(frozen=True)
class HfAuth(ObjectLocation):
    token: str


@dataclass(frozen=True)
class HfModelLocation(ExternalAuth):
    repo_id: str
    filename: str



