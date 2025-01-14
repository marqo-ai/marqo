from typing import Any, Dict, Optional

from pydantic import BaseModel


class VespaDocument(BaseModel):
    id: Optional[str]
    create_timestamp: Optional[float]
    field_types: Optional[Dict[str, str]]
    fields: Dict[str, Any]
