from model_management.schemas.api_models import LoadModelRequest
from model_management.schemas.triton_model_properties import (
    TritonModelProperties,
    ModelOutput,
    ModelInput,
)
from unittest import TestCase


class DummyTest(TestCase):
    def test_load_model_request_serialization(self):
        """A dummy test to trigger the integration tests."""
        _ = LoadModelRequest(
            triton_model_properties=TritonModelProperties(
                name="test",
                max_batch_size=8,
                sources=["s3://bucket/model.onnx"],
                input=[
                    ModelInput(
                        name="input_1", dims=[-1, 3, 224, 224], data_type="TYPE_FP32"
                    )
                ],
                output=[
                    ModelOutput(name="output_1", dims=[-1, 1000], data_type="TYPE_FP32")
                ],
            )
        )
