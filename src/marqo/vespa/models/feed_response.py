from typing import Optional, List

from pydantic import BaseModel, Field


class FeedDocumentResponse(BaseModel):
    """A response from feeding a document to Vespa."""
    status: int
    path_id: str = Field(alias='pathId')


class FeedBatchDocumentResponse(BaseModel):
    """A response from feeding a document to Vespa while the
    document is part of a batch."""
    status: int
    path_id: str = Field(alias='pathId')
    id: Optional[str]
    message: Optional[str]


class FeedBatchResponse(BaseModel):
    """A response from feeding a batch of documents to Vespa."""
    responses: List[FeedBatchDocumentResponse]
    errors: bool
