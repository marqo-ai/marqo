from marqo.base_model import StrictBaseModel
from typing import List


class MarqoCudaInfo(StrictBaseModel):
    device_id: int
    device_name: str
    memory_used: str
    total_memory: str
    utilization: str
    memory_used_percent: str


class MarqoCudaInfoResponse(StrictBaseModel):
    cuda_devices: List[MarqoCudaInfo]
