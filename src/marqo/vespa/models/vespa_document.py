from pydantic import BaseModel
from typing import Any, Dict


class VespaDocument(BaseModel):
    id: str
    fields: Dict[str, Any]
