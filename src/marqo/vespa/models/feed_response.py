from typing import Optional, List

from pydantic import BaseModel, Field


class FeedDocumentResponse(BaseModel):
    path_id: str = Field(alias='pathId')
    id: str


class FeedBatchDocumentResponse(BaseModel):
    status: int
    path_id: str = Field(alias='pathId')
    id: Optional[str]
    message: Optional[str]


class FeedBatchResponse(BaseModel):
    responses: List[FeedBatchDocumentResponse]
    errors: bool
