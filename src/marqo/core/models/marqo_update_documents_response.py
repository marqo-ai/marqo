from typing import List, Optional

from pydantic import Field

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
    _success_count: Field(exclude=True, default=0)
    _error_count: Field(exclude=True, default=0)
