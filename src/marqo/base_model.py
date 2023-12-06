from pydantic import BaseModel


class StrictBaseModel(BaseModel):
    class Config:
        extra: str = "forbid"
        validate_assignment: bool = True


class ImmutableBaseModel(BaseModel):
    class Config:
        allow_mutation: bool = False


class ImmutableStrictBaseModel(StrictBaseModel, ImmutableBaseModel):
    class Config(StrictBaseModel.Config, ImmutableBaseModel.Config):
        pass
