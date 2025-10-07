from model_management.schemas.app_models import AppBaseModel
from pydantic import Field
from model_management.schemas.triton_model_properties import TritonModelProperties


class LoadModelRequest(AppBaseModel):
    triton_model_properties: TritonModelProperties = Field(..., validation_alias='tritonModelProperties')