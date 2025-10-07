from pydantic import BaseModel, Field
from typing import Any, Optional

class Problem(BaseModel):
    type: str = "about:blank"
    title: str
    status: int
    code: str
    detail: Optional[str] = None
    instance: Optional[str] = None
    request_id: Optional[str] = None
    extras: Optional[dict[str, Any]] = Field(default=None)
