"""This regards structuring information regrading customer-stored ML models

For example models stored on custom Huggingface repos or on private s3 buckets
"""
from marqo.tensor_search.models.external_apis.hf import HfAuth, HfModelLocation
from marqo.tensor_search.models.external_apis.s3 import S3Auth, S3Location
from pydantic import BaseModel, validator
from marqo.errors import InvalidArgError
from typing import Optional

class ModelAuth(BaseModel):
    """TODO: insert links to docs in error message"""
    class Config:
        allow_mutation = False

    s3: Optional[S3Auth] = None
    hf: Optional[HfAuth] = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.s3 is None and self.hf is None:
            raise InvalidArgError(
                "Missing authentication object. An authentic object, for example `s3` or  "
                "`hf`, must be provided. ")

    @validator('s3', 'hf', pre=True, always=True)
    def _ensure_exactly_one_auth_method(cls, v, values, field):
        other_field = 's3' if field.name == 'hf' else 'hf'
        if other_field in values and values[other_field] is not None and v is not None:
            raise InvalidArgError(
                "More than one model authentication was provided. "
                "Only one model authentication object is allowed")
        return v


class ModelLocation(BaseModel):

    class Config:
        allow_mutation = False

    s3: Optional[S3Location] = None
    hf: Optional[HfModelLocation] = None
    auth_required: bool = False

    @validator('s3', 'hf', pre=True, always=True)
    def _ensure_exactly_one_location(cls, v, values, field):
        """TODO: insert links to docs in error message"""
        other_field = 's3' if field.name == 'hf' else 'hf'
        if other_field in values and values[other_field] is not None and v is not None:
            raise InvalidArgError(
                "More than one model location object was provided. "
                "Only one model authentication object is allowed")
        return v

    def __init__(self, **data):
        super().__init__(**data)
        if self.s3 is None and self.hf is None:
            raise InvalidArgError(
                "Missing model location object. A location object, for example `s3` or  "
                "`hf`, must be provided. ")

