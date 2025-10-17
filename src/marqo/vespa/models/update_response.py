from typing import List, Optional

from pydantic.v1 import BaseModel, Field


class UpdateDocumentResponse(BaseModel):
    status: int
    path_id: Optional[str] = Field(default=None, alias="pathId")
    id: Optional[str]
    message: Optional[str]


class UpdateDocumentsBatchResponse(BaseModel):
    responses: List[UpdateDocumentResponse]
    errors: bool
