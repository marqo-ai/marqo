from marqo_model_management_container.schemas.app_models import AppBaseModel
from pydantic import Field
from marqo_model_management_container.schemas.triton_model_properties import TritonModelProperties


class LoadModelRequest(AppBaseModel):
    triton_model_properties: TritonModelProperties = Field(..., validation_alias='tritonModelProperties')