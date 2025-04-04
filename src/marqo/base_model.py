from pydantic import BaseModel, ConfigDict


class MarqoBaseModel(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,  # accept both real name and alias (if present)
        validate_assignment=True
    )


class StrictBaseModel(MarqoBaseModel):
    model_config = ConfigDict(
        **MarqoBaseModel.model_config,
        extra="forbid"
    )


class ImmutableBaseModel(MarqoBaseModel):
    model_config = ConfigDict(
        **MarqoBaseModel.model_config,
        frozen=True
    )


class ImmutableStrictBaseModel(StrictBaseModel, ImmutableBaseModel):
    model_config = ConfigDict(
        **StrictBaseModel.model_config,
        **{k: v for k, v in ImmutableBaseModel.model_config.items() if k != "populate_by_name" and k != "validate_assignment"}
    )