from pydantic.dataclasses import dataclass
from typing import Optional
from marqo.tensor_search.models.external_apis.abstract_classes import (
    ObjectLocation, ExternalAuth
)


class S3Auth(ExternalAuth):
    aws_secret_access_key: str
    aws_access_key_id: str
    aws_session_token: Optional[str] = None


@dataclass(frozen=True)
class S3Location(ObjectLocation):
    Bucket: str
    Key: str

