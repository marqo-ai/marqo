from pydantic import BaseModel, ConfigDict


class AppBaseModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
    )


class AppStrBaseModel(AppBaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid"
    )