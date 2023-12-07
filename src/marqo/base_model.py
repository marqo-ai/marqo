from pydantic import BaseModel


class MarqoBaseModel(BaseModel):
    class Config:
        # allow_population_by_field_name = True  # deserialize both real name and alias (if present)
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
