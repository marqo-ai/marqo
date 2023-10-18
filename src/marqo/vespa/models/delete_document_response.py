from typing import Optional

from pydantic import BaseModel


class DeleteDocumentResponse(BaseModel):
    status: str
    pathId: str
    id: Optional[str]
