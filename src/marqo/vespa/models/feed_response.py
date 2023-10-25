from typing import Optional, List

from pydantic import BaseModel


class FeedResponse(BaseModel):
    status: str
    pathId: str
    id: Optional[str]
    message: Optional[str]


class FeedBatchResponse(BaseModel):
    responses: List[FeedResponse]
    errors: bool


class FeedDocumentResponse(BaseModel):
    pathId: str
    id: str
