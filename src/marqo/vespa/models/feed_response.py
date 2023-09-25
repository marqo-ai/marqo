from typing import Optional, List

from pydantic import BaseModel


class FeedResponse(BaseModel):
    status: str
    pathId: str
    id: Optional[str]
    message: Optional[str]


class BatchFeedResponse(BaseModel):
    responses: List[FeedResponse]
    errors: bool
