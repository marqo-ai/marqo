from typing import Optional

from marqo.core.models.strict_base_model import StrictBaseModel


class VespaStats(StrictBaseModel):
    memory_used_percentage: Optional[float]
    storage_used_percentage: Optional[float]


class MarqoIndexStats(StrictBaseModel):
    number_of_documents: int
    number_of_vectors: int
    backend: VespaStats
