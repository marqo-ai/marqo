from typing import Optional

from marqo.base_model import StrictBaseModel


class InferenceHealthResponse(StrictBaseModel):
    status: str


class BackendHealthResponse(StrictBaseModel):
    status: str
    memoryIsAvailable: Optional[bool]
    storageIsAvailable: Optional[bool]


class HealthResponse(StrictBaseModel):
    status: str
    inference: InferenceHealthResponse
    backend: BackendHealthResponse

    @classmethod
    def from_marqo_health_status(cls, marqo_health_status):
        return cls(
            status=marqo_health_status.status.value,
            inference=InferenceHealthResponse(
                status=marqo_health_status.inference.status.value
            ),
            backend=BackendHealthResponse(
                status=marqo_health_status.backend.status.value,
                memoryIsAvailable=marqo_health_status.backend.memory_is_available,
                storageIsAvailable=marqo_health_status.backend.storage_is_available
            )
        )
