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
    index_name: str
    items: List[MarqoAddDocumentsItem]

    @root_validator(pre=True)
    def check_errors(cls, values):
        items = values.get('items')
        errors = False
        for item in items:
            if item.error:
                errors = True
                break
        values["errors"] = errors
        return values