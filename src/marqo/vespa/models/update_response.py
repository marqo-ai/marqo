from typing import Optional, List

from pydantic import BaseModel, Field


class UpdateDocumentResponse(BaseModel):
    path_id: str = Field(alias='pathId')
    id: str


class UpdateBatchDocumentResponse(BaseModel):
    status: int
    path_id: str = Field(alias='pathId')
    id: Optional[str]
    message: Optional[str]


class UpdateBatchResponse(BaseModel):
    responses: List[UpdateBatchDocumentResponse]
    errors: bool
