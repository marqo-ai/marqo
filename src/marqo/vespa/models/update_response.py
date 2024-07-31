from typing import Optional, List

from pydantic import BaseModel, Field, root_validator


class UpdateDocumentResponse(BaseModel):
    status: int
    path_id: Optional[str] = Field(default=None, alias='pathId')
    id: Optional[str]
    message: Optional[str]


class UpdateDocumentsBatchResponse(BaseModel):
    responses: List[UpdateDocumentResponse]
    errors: bool
