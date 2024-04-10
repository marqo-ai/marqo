from typing import Optional, List

from pydantic import BaseModel, Field, root_validator


class UpdateDocumentResponse(BaseModel):
    status: int
    path_id: str = Field(alias='pathId')
    id: Optional[str]
    message: Optional[str]

    @root_validator(pre=True)
    def check_status_and_message(cls, values):
        status = values.get('status')
        if status == 412:
            values["message"] = "Document does not exist in the index."
            values["status"] = 404
        return values


class UpdateDocumentsBatchResponse(BaseModel):
    responses: List[UpdateDocumentResponse]
    errors: bool
