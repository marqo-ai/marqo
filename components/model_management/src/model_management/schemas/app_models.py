from pydantic import BaseModel, ConfigDict


class AppBaseModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )


class AppStrBaseModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class AppImmutableBaseModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, frozen=True)
