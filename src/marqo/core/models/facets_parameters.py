from enum import Enum
from typing import List, Optional, Literal, Dict

from marqo.base_model import StrictBaseModel

class FacetsParameters(StrictBaseModel):
    facetFields: Optional[List[str]] = None
    maxDepth: Optional[int] = None
    maxResults: Optional[int] = None
    order: Literal["ASC", "DESC"] = "DESC"
