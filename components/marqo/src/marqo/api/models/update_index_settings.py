from pydantic import Field

from marqo.base_model import MarqoStrictModelV2


class UpdateIndexSettingsBodyParams(MarqoStrictModelV2):
    """Model for the body parameters of the update_index_settings endpoint.

    Currently, only updating model_properties is supported.
    """
    model_properties: dict = Field(default_factory=dict, alias="modelProperties")