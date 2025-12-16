from model_management.schemas.app_models import AppBaseModel
from model_management.schemas.triton_model_properties import TritonModelProperties
from pydantic import Field


class LoadModelRequest(AppBaseModel):
    triton_model_properties: TritonModelProperties = Field(
        ..., validation_alias="tritonModelProperties"
    )


class LoadModelResponse(AppBaseModel):
    message: str = Field(
        ..., description="A message indicating the result of the load model operation."
    )


class UnloadModelResponse(AppBaseModel):
    message: str = Field(
        ...,
        description="A message indicating the result of the unload model operation.",
    )
