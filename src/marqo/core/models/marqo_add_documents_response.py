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

    @root_validator(pre=True)
    def check_status_and_message(cls, values):
        status = values.get('status')
        message = values.get('message')
        if not isinstance(status, int):
            raise ValueError(f"status must be an integer, got {status}")
        if status == 200:
            return values
        elif status == 429:
            values["error"] = "Marqo vector store receives too many requests. Please try again later."
        elif status == 507:
            values["error"] = "Marqo vector store is out of memory or disk space."
        else:
            values["status"] = 500
            values["error"] = message
        return values


class MarqoAddDocumentsResponse(MarqoBaseModel):
    errors: bool
    processingTimeMs: float
    index_name: str # TODO Change this to camelCase in the future (Breaking change!)
    items: List[MarqoAddDocumentsItem]