from typing import List, Optional

from pydantic import Field, root_validator

from marqo.base_model import MarqoBaseModel


class MarqoUpdateDocumentsItem(MarqoBaseModel):
    id: Optional[str] = Field(alias="_id", default=None)
    status: int
    message: Optional[str] = None
    error: Optional[str] = None

    @root_validator(pre=True)
    def check_status_and_message(cls, values):
        status = values.get('status')
        message = values.get('message')
        if status == 412:
            values["message"] = "Document does not exist in the index."
            values["status"] = 404
        if isinstance(status, int) and status > 400 and message:
            values["error"] = message
        return values


class MarqoUpdateDocumentsResponse(MarqoBaseModel):
    errors: bool
    index_name: str
    items: List[MarqoUpdateDocumentsItem]
    processingTimeMs: float