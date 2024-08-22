from typing import List, Optional

from pydantic import Field, root_validator

from marqo.base_model import MarqoBaseModel
from marqo.core.models.marqo_add_documents_response import BatchResponseStats
from marqo.core.models.marqo_add_documents_response import MarqoBaseDocumentsResponse


class MarqoUpdateDocumentsItem(MarqoBaseModel):
    id: Optional[str] = Field(alias="_id", default=None)
    status: int
    message: Optional[str] = None
    error: Optional[str] = None


class MarqoUpdateDocumentsResponse(MarqoBaseDocumentsResponse):
    errors: bool
    index_name: str
    items: List[MarqoUpdateDocumentsItem]
    processingTimeMs: float

    @root_validator(pre=False, skip_on_failure=True)
    def count_items(cls, values):
        items = values.get("items")
        batch_response_count = BatchResponseStats()

        if items:
            for item in items:
                if item.status in range(200, 300):
                    batch_response_count.success_count += 1
                elif item.status in range(400, 500):
                    batch_response_count.failure_count += 1
                elif item.status >= 500:
                    batch_response_count.error_count += 1
                else:
                    raise ValueError(f"Unexpected status code: {item.status}")

        values['_batch_response_stats'] = batch_response_count
        return values