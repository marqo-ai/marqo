from typing import List, Optional

from pydantic import BaseModel, Field


class DeleteDocumentResponse(BaseModel):
    path_id: str = Field(alias='pathId')
    id: str


class DeleteBatchDocumentResponse(BaseModel):
    status: int
    path_id: str = Field(alias='pathId')
    id: Optional[str]
    message: Optional[str]


class DeleteBatchResponse(BaseModel):
    responses: List[DeleteBatchDocumentResponse]
    errors: bool


class DeleteAllDocumentsResponse(BaseModel):
    path_id: str = Field(alias='pathId')
    document_count: int = Field(alias='documentCount')