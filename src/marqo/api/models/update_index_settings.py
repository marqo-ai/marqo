from typing import Dict, Any

import pydantic.v1

from marqo.base_model import StrictBaseModel


class UpdateIndexSettingsBodyParams(StrictBaseModel):
    model_properties: Dict[str, Any] = pydantic.v1.Field(alias='modelProperties')
