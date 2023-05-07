"""This regards structuring information regrading customer-stored ML models

For example models stored on custom Huggingface repos or on private s3 buckets
"""

from typing import Optional
from pydantic import BaseModel
from marqo.tensor_search.models.external_apis.hf import HfAuth, HfModelLocation
from marqo.tensor_search.models.external_apis.s3 import S3Auth, S3Location
from marqo.errors import InvalidArgError


class ModelAuth(BaseModel):
    class Config:
        allow_mutation = False

    s3: Optional[S3Auth] = None
    hf: Optional[HfAuth] = None

    def __post_init__(self):
        self._ensure_exactly_one_auth_method()

    def _ensure_exactly_one_auth_method(self):
        """TODO: insert links to docs in error message"""
        auth_objects = [self.s3, self.hf]

        auth_objects_presence = [x is not None for x in auth_objects]

        if not any(auth_objects_presence):
            raise InvalidArgError(
                "Missing authentication object. An authentic object, for example `s3` or  "
                "`hf`, must be provided. ")
        if sum(auth_objects_presence) > 1:
            raise InvalidArgError(
                "More than one model authentication was provided. "
                "Only one model authentication object is allowed")


class ModelLocation(BaseModel):

    class Config:
        allow_mutation = False

    s3: Optional[S3Location] = None
    hf: Optional[HfModelLocation] = None
    auth_required: bool = False

    def __post_init__(self):
        self._ensure_exactly_one_location()

    def _ensure_exactly_one_location(self):
        """TODO: insert links to docs in error message"""
        loc_objects = [self.s3, self.hf]

        loc_objects_presence = [x is not None for x in loc_objects]

        if not any(loc_objects_presence):
            raise InvalidArgError(
                "Missing model location object. A location object, for example `s3` or  "
                "`hf`, must be provided. ")
        if sum(loc_objects_presence) > 1:
            raise InvalidArgError(
                "More than one model location object was provided. "
                "Only one model authentication object is allowed")
