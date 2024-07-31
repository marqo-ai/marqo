from typing import List, Optional

from pydantic import Field, root_validator

from marqo.base_model import MarqoBaseModel


class MarqoAddDocumentsItem(MarqoBaseModel):
    """A response from adding a document to Marqo.

    This model takes the response from Marqo vector store and translate it to a user-friendly response.
    """
    status: int
    id: Optional[str] = Field(alias="_id", default=None)
    message: Optional[str] = None
    error: Optional[str] = None
    code: Optional[str] = None


class MarqoAddDocumentsResponse(MarqoBaseModel):
    errors: bool
    processingTimeMs: float
    index_name: str  # TODO Change this to camelCase in the future (Breaking change!)
    items: List[MarqoAddDocumentsItem]
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