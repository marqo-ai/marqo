from typing import Any, Dict, Optional

from pydantic import BaseModel


class VespaDocument(BaseModel):
    id: Optional[str]
    create_timestamp: Optional[float]
    fields: Dict[str, Any]
