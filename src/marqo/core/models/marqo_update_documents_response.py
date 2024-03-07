from typing import List, Optional

from pydantic import Field, root_validator

from marqo.base_model import MarqoBaseModel


class MarqoUpdateDocumentsItem(MarqoBaseModel):
    id: str = Field(alias="_id")
    status: int
    message: Optional[str] = None
    error: Optional[str] = None

    @root_validator(pre=True)
    def check_status_and_message(cls, values):
        status = values.get('status')
        message = values.get('message')
        if status >= 400 and message:
            values["error"] = message
        return values


class MarqoUpdateDocumentsResponse(MarqoBaseModel):
    errors: bool
    indexName: str = Field(alias="index_name")
    items: List[MarqoUpdateDocumentsItem]
    preprocessingTimeMs: float

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