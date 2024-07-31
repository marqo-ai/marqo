from typing import List, Optional

from pydantic import Field, root_validator

from marqo.base_model import MarqoBaseModel


class MarqoUpdateDocumentsItem(MarqoBaseModel):
    id: Optional[str] = Field(alias="_id", default=None)
    status: int
    message: Optional[str] = None
    error: Optional[str] = None


class MarqoUpdateDocumentsResponse(MarqoBaseModel):
    errors: bool
    index_name: str
    items: List[MarqoUpdateDocumentsItem]
    processingTimeMs: float

    _success_count: int = Field(exclude=True, default=0)
    _error_count: int = Field(exclude=True, default=0)
    _fail_count: int = Field(exclude=True, default=0)

    @root_validator(pre=False, skip_on_failure=True)
    def count_items(cls, values):
        items = values.get("items")
        if items:
            for item in items:
                if item.status in range(200, 300):
                    values["_success_count"] += 1
                elif item.status in range(400, 500):
                    values["_fail_count"] += 1
                elif item.status >= 500:
                    values["_error_count"] += 1
                else:
                    raise ValueError(f"Unexpected status code: {item.status}")
        return values
