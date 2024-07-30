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