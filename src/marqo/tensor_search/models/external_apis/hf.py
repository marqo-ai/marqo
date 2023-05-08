from pydantic.dataclasses import dataclass
from marqo.tensor_search.models.external_apis.abstract_classes import (
    ObjectLocation, ExternalAuth
)


class HfAuth(ObjectLocation):
    token: str


class HfModelLocation(ExternalAuth):
    repo_id: str
    filename: str



