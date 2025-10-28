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


"""
The configuration propagation behaviour can be found here:
https://docs.pydantic.dev/latest/concepts/config/#change-behaviour-globally

TLDR, If you wish to change the behaviour of Pydantic globally,
you can create your own custom parent class with a custom configuration, as the configuration is inherited.
If you provide configuration to the subclasses, it will be merged with the parent configuration.

NOTE: If your model inherits from multiple bases, Pydantic currently doesn't follow the MRO! Avoid this for
unexpected behaviour.
"""


class MarqoBaseModelV2(pydantic.BaseModel):
    model_config = ConfigDict(validate_by_name=True, validate_assignment=True)


class StrictBaseModelV2(MarqoBaseModelV2):
    model_config = ConfigDict(extra="forbid")


class ImmutableBaseModelV2(MarqoBaseModelV2):
    model_config = ConfigDict(frozen=True)


class ImmutableStrictBaseModelV2(ImmutableBaseModelV2):
    model_config = ConfigDict(extra="forbid")