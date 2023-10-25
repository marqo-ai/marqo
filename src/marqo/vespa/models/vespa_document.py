from typing import Any, Dict, Optional

from pydantic import BaseModel


class VespaDocument(BaseModel):
    id: Optional[str]
    fields: Dict[str, Any]
