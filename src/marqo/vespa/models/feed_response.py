from typing import Optional, List

from pydantic import BaseModel, Field


class FeedDocumentResponse(BaseModel):
    """A response from feeding a document to Vespa."""
    status: int
    path_id: str = Field(alias='pathId')
    id: Optional[str]
    message: Optional[str]


class FeedBatchResponse(BaseModel):
    """A response from feeding a batch of documents to Vespa."""
    responses: List[FeedDocumentResponse]
    errors: bool
