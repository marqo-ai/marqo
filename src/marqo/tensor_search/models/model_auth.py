from typing import Optional
from pydantic.dataclasses import dataclass
from pydantic import validator
from marqo.tensor_search.models.external_apis.hf import HfAuth
from marqo.tensor_search.models.external_apis.s3 import S3Auth
from marqo.errors import InvalidArgError


@dataclass(frozen=True)
class ModelAuth:
    s3: Optional[S3Auth] = None
    hf: Optional[HfAuth] = None

    @validator("hf", "s3", always=True)
    def validate_auth(cls, value, values, field):
        """TODO: insert links to docs in error message"""
        auth_objects = [values.get("s3"), values.get("hf")]

        auth_objects_presence = [x is None for x in auth_objects]

        if not any(auth_objects_presence):
            raise InvalidArgError(
                "Missing authentication object. An authentic object, for example `s3` or  "
                "`hf`, must be provided. ")
        if sum(auth_objects_presence) > 1:
            raise InvalidArgError(
                "More than one model authentication was provided. "
                "Only one model authentication object is allowed")
        return value
