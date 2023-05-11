from pydantic.dataclasses import dataclass
from marqo.tensor_search.models.external_apis.abstract_classes import (
    ObjectLocation, ExternalAuth
)


class HfAuth(ExternalAuth):
    token: str


class HfModelLocation(ObjectLocation):
    repo_id: str
    filename: str



