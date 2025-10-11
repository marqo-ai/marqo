import pydantic
from pydantic import ConfigDict
from pydantic.v1 import BaseModel


class MarqoBaseModel(BaseModel):
    class Config:
        allow_population_by_field_name = True  # accept both real name and alias (if present)
        validate_assignment = True


class StrictBaseModel(MarqoBaseModel):
    class Config(MarqoBaseModel.Config):
        extra = "forbid"


class ImmutableBaseModel(MarqoBaseModel):
    class Config(MarqoBaseModel.Config):
        allow_mutation = False


class ImmutableStrictBaseModel(StrictBaseModel, ImmutableBaseModel):
    class Config(StrictBaseModel.Config, ImmutableBaseModel.Config):
        pass


class MarqoBaseModelV2(pydantic.BaseModel):
    model_config = ConfigDict(validate_by_name=True, validate_assignment=True)


class MarqoStrictModelV2(MarqoBaseModelV2):
    model_config = ConfigDict(**MarqoBaseModelV2.model_config, extra="forbid")