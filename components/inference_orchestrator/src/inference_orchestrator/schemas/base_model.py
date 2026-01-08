from pydantic import BaseModel, ConfigDict


class AppBaseModel(BaseModel):
    model_config = ConfigDict(
        validate_by_alias=True,
        validate_by_name=True,
        extra="ignore",
    )


class AppImmutableBaseModel(BaseModel):
    model_config = ConfigDict(
        validate_by_name=True, validate_by_alias=True, extra="ignore", frozen=True
    )
