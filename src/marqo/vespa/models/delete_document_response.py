from typing import Optional

from pydantic import BaseModel


class DeleteDocumentResponse(BaseModel):
    pathId: str
    id: str
