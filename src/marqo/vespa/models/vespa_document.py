from typing import Any, Dict, Optional

from pydantic import BaseModel


class VespaDocument(BaseModel):
    id: Optional[str]
    field_types: Optional[Dict[str, str]] # A metadata field to store the type of each field in the document
    fields: Dict[str, Any]
    create_timestamp: Optional[float]
